import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from datasets.palettes import get_palette

from torchmetrics import F1Score

from third_party.unimatch.dataset.semi import SemiDataset
from model.builder import build_model
from third_party.unimatch.supervised import predict
from datasets.classes import CLASSES
from third_party.unimatch.util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from third_party.unimatch.util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--ema', action='store_true')
parser.add_argument('--pred-path', default=None, type=str)
parser.add_argument('--logit-path', default=None, type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, mode, cfg, distributed=True, pred_path=None, logit_path=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    palette = get_palette(cfg['dataset'])
    
    # int f1 metric
    f1_metric = F1Score(
        task='multiclass',
        num_classes=cfg['nclass'], 
        average='macro',
        ignore_index=255  
    ).cuda()
    
    # per class f1 scores
    f1_per_class_metric = F1Score(
        task='multiclass',
        num_classes=cfg['nclass'], 
        average=None,
        ignore_index=255
    ).cuda()

    with torch.no_grad():
        did_update_f1 = False
        did_update_iou = False
        for img, mask, id in tqdm(loader, total=len(loader)):
            file_name, lbl_name = id[0].split(' ')
            img = img.cuda()
            mask = mask.cuda()  

            pred, final = predict(model, img, mask, mode, cfg, return_logits=True)

            if logit_path is not None:
                logit_file = os.path.join(logit_path, lbl_name.split('/')[-1])\
                    .replace('.png', '.pt')
                os.makedirs(os.path.dirname(logit_file), exist_ok=True)
                torch.save(final.detach().cpu(), logit_file)

            if pred_path is not None:
                pred_file = os.path.join(pred_path, lbl_name.split('/')[-1])
                os.makedirs(os.path.dirname(pred_file), exist_ok=True)
                np_pred = pred[0].cpu().numpy().astype(np.uint8)
                output = Image.fromarray(np_pred).convert('P')
                output.putpalette(palette)
                output.save(pred_file)


            if isinstance(pred, np.ndarray):
                pred_tensor = torch.from_numpy(pred).cuda()
            else:
                pred_tensor = pred.cuda() if not pred.is_cuda else pred

            # Ensure prediction is class indices [B,H,W]
            if pred_tensor.dim() == 4 and pred_tensor.size(1) == cfg['nclass']:
                pred_tensor = pred_tensor.argmax(dim=1)

            # Ensure both are [B,H,W]
            if mask.dim() == 4 and mask.size(-1) == 3:
                # RGB mask is incompatible for metrics; skip F1/IoU but still allow saving preds
                compute_metrics = False
            else:
                compute_metrics = True

            if compute_metrics:
                if pred_tensor.shape != mask.shape:
                    # Try to align spatial dims only
                    b = min(pred_tensor.shape[0], mask.shape[0]) if pred_tensor.dim() > 0 else 1
                    pred_tensor = pred_tensor.view(b, *pred_tensor.shape[-2:])
                    mask_tensor = mask.view(b, *mask.shape[-2:])
                    if pred_tensor.shape != mask_tensor.shape:
                        pred_tensor = torch.nn.functional.interpolate(
                            pred_tensor.unsqueeze(1).float(),
                            size=mask_tensor.shape[-2:],
                            mode='nearest'
                        ).squeeze(1).long()
                else:
                    mask_tensor = mask

                # Update F1 on flattened vectors
                f1_metric.update(pred_tensor.flatten(), mask_tensor.flatten())
                f1_per_class_metric.update(pred_tensor.flatten(), mask_tensor.flatten())
                did_update_f1 = True


            if compute_metrics:
                pred_np = pred.cpu().numpy()
                mask_np = mask.cpu().numpy()
                # Squeeze RGB masks to single channel if present
                if mask_np.ndim == 4 and mask_np.shape[-1] == 3:
                    mask_np = mask_np[..., 0]
                intersection, union, target = intersectionAndUnion(
                    pred_np, mask_np, cfg['nclass'], 255
                )

                reduced_intersection = torch.from_numpy(intersection).cuda()
                reduced_union = torch.from_numpy(union).cuda()
                reduced_target = torch.from_numpy(target).cuda()

                if distributed:
                    dist.all_reduce(reduced_intersection)
                    dist.all_reduce(reduced_union)
                    dist.all_reduce(reduced_target)

                intersection_meter.update(reduced_intersection.cpu().numpy())
                union_meter.update(reduced_union.cpu().numpy())
                did_update_iou = True


    if did_update_iou:
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        mIOU = np.mean(iou_class)
    else:
        iou_class = np.array([])
        mIOU = float('nan')
    

    if did_update_f1:
        mean_f1 = f1_metric.compute()
        f1_per_class = f1_per_class_metric.compute()
    else:
        mean_f1 = torch.tensor(float('nan')).cuda()
        f1_per_class = torch.tensor([]).cuda()
    
    
    f1_metric.reset()
    f1_per_class_metric.reset()

    return mIOU, iou_class, mean_f1, f1_per_class

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    cfg.setdefault('text_embedding_variant', None)
    cfg.setdefault('pl_text', cfg['text_embedding_variant'])
    cfg['clip_encoder'] = None

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    if args.port is not None:
        rank, world_size = setup_distributed(port=args.port)
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size = 0, 1
        local_rank = 0

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

    cudnn.enabled = True
    cudnn.benchmark = True

    model = build_model(cfg)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    if args.port is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                        output_device=local_rank, find_unused_parameters=('zegclip' in cfg['model']))
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        valsampler = None

    valset = SemiDataset(cfg, 'val')

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    if args.save_path != 'none':
        checkpoint = torch.load(os.path.join(args.save_path))
        if args.ema:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['ema_model'].items()}
        else:
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        for k in list(checkpoint['model'].keys()):
            if 'clip_encoder' in k:
                del checkpoint['model'][k]
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    else:
        if rank == 0:
            logger.info('************ WARNING: NO CHECKPOINT SPECIFIED')
    

    if 'eval_mode' in cfg:
        eval_mode = cfg['eval_mode']
    else:
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    
    mIoU, iou_class, mean_f1, f1_per_class = evaluate(
        model, valloader, eval_mode, cfg, 
        distributed=args.port is not None,
        pred_path=args.pred_path,
        logit_path=args.logit_path)

    if rank == 0:
        # iou
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
        
        # f1
        for (cls_idx, f1) in enumerate(f1_per_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'F1: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], f1.item() * 100))
        logger.info('***** Evaluation {} ***** >>>> Mean F1: {:.2f}\n'.format(eval_mode, mean_f1.item() * 100))


if __name__ == '__main__':
    main()

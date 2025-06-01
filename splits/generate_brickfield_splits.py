import os
import random

# Paths
# find data directory by looking up from current location
current_dir = os.getcwd()
while current_dir != '/':
    potential_data = os.path.join(current_dir, 'data/BrickField_512')
    if os.path.exists(potential_data):
        data_root = potential_data + '/'
        break
    current_dir = os.path.dirname(current_dir)
else:
    data_root = '/app/data/BrickField_512/'  # fallback
train_img_dir = os.path.join(data_root, "train/images")
train_gt_dir = os.path.join(data_root, "train/gts")
test_img_dir = os.path.join(data_root, "test/images")
test_gt_dir = os.path.join(data_root, "test/gts")

output_root = "brickfield"
os.makedirs(output_root, exist_ok=True)

# Total training images
total_train = 26986

# Calculate split sizes (1%, 2%, 4%, 8%, 20%)
split_percentages = [0.01, 0.02, 0.04, 0.08, 0.20]
split_sizes = [int(total_train * pct) for pct in split_percentages]

print(f"Total training images: {total_train}")
print("Split sizes:")
for i, (pct, size) in enumerate(zip(split_percentages, split_sizes)):
    print(f"  {pct * 100:4.0f}%: {size:5d} labeled, {total_train - size:5d} unlabeled")

# Read and shuffle train image filenames
all_imgs = sorted([f for f in os.listdir(train_img_dir) if f.endswith(".png")])
print(f"\nActual images found: {len(all_imgs)}")

if len(all_imgs) != total_train:
    print(f"WARNING: Expected {total_train} images but found {len(all_imgs)}")

random.seed(42)  # Changed seed for reproducibility
random.shuffle(all_imgs)


def save_txt(path, entries, split_type="train"):
    """Save image paths to text file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for e in entries:
            img_name = e.replace(".png", "")
            if split_type == "train":
                f.write(f"train/images/{img_name}.png train/gts/{img_name}_gt.png\n")
            else:  # test/val
                f.write(f"test/images/{img_name}.png test/gts/{img_name}_gt.png\n")


# Create splits
for split_size in split_sizes:
    labeled = all_imgs[:split_size]
    unlabeled = all_imgs[split_size:]

    folder = os.path.join(output_root, str(split_size))

    print(f"\nCreating split {split_size}:")
    print(f"  Labeled: {len(labeled)} images")
    print(f"  Unlabeled: {len(unlabeled)} images")

    save_txt(os.path.join(folder, "labeled.txt"), labeled)
    save_txt(os.path.join(folder, "unlabeled.txt"), unlabeled)

# Save test/validation set
test_imgs = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".png")])
print(f"\nTest images found: {len(test_imgs)}")

save_txt(os.path.join(output_root, "val.txt"), test_imgs, split_type="test")

print(f"\nAll splits saved to: {output_root}")
print("\nGenerated splits:")
for split_size in split_sizes:
    pct = (split_size / total_train) * 100
    print(f"  {split_size:4d} ({pct:4.1f}%)")
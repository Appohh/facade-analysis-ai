import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Define your dataset directory
dataset_dir = './generated_facades'  # Change this to your actual folder path

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Collect all image-annotation pairs
image_files = sorted([f for f in os.listdir(dataset_dir) if f.startswith('facade_') and f.endswith('.png')])
annotation_files = sorted([f for f in os.listdir(dataset_dir) if f.startswith('mask_') and f.endswith('.png')])

# Extract the numerical part of the filenames to match images and masks
pairs = []
for img in image_files:
    num = img.split('_')[1].split('.')[0]
    mask = f'mask_{num}.png'
    if mask in annotation_files:
        pairs.append((img, mask, num))
    else:
        print(f"Warning: No annotation found for {img}")

# Shuffle the pairs
random.shuffle(pairs)

# Compute split sizes
total_pairs = len(pairs)
train_size = int(train_ratio * total_pairs)
val_size = int(val_ratio * total_pairs)
test_size = total_pairs - train_size - val_size  # Ensure all are used

# Split the dataset
train_pairs = pairs[:train_size]
val_pairs = pairs[train_size:train_size + val_size]
test_pairs = pairs[train_size + val_size:]

# Define output folders
output_folders = [
    'images/train', 'images/val', 'images/test',
    'annotations/train', 'annotations/val', 'annotations/test'
]

for folder in output_folders:
    os.makedirs(os.path.join(dataset_dir, folder), exist_ok=True)

# Helper function to copy and rename files
def copy_and_rename(pairs, split):
    for idx, (img_file, mask_file, num) in enumerate(pairs):
        # Rename: facade_xxx.png -> image_xxx.png
        new_img_name = f'image_{num}.png'
        new_mask_name = f'annotation_{num}.png'
        
        shutil.copy(
            os.path.join(dataset_dir, img_file),
            os.path.join(dataset_dir, f'images/{split}', new_img_name)
        )
        shutil.copy(
            os.path.join(dataset_dir, mask_file),
            os.path.join(dataset_dir, f'annotations/{split}', new_mask_name)
        )

# Copy and rename the files
copy_and_rename(train_pairs, 'train')
copy_and_rename(val_pairs, 'val')
copy_and_rename(test_pairs, 'test')

print("Dataset split and renaming complete!")

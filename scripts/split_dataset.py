import os
import random
import shutil

# Set paths
base_dir = './nelissen_dataset'
images_dir = os.path.join(base_dir, 'images')
annotations_dir = os.path.join(base_dir, 'annotations')

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Set seed for reproducibility
random.seed(42)

# Get all image filenames
image_files = sorted([f for f in os.listdir(images_dir) if f.startswith('image_') and f.endswith('.png')])
annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.startswith('annotation_') and f.endswith('.png')])

# Extract numeric ID and match pairs
image_ids = [f.split('_')[1].split('.')[0] for f in image_files]
annotation_ids = [f.split('_')[1].split('.')[0] for f in annotation_files]

# Ensure matching IDs
matched_ids = sorted(set(image_ids) & set(annotation_ids))

print(f"✅ Found {len(matched_ids)} matching image-annotation pairs.")

# Shuffle IDs
random.shuffle(matched_ids)

# Split
total = len(matched_ids)
train_split = int(train_ratio * total)
val_split = int(val_ratio * total)

train_ids = matched_ids[:train_split]
val_ids = matched_ids[train_split:train_split + val_split]
test_ids = matched_ids[train_split + val_split:]

splits = {
    'train': train_ids,
    'val': val_ids,
    'test': test_ids
}

# Create output folders
for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(annotations_dir, split), exist_ok=True)

# Copy files
for split, ids in splits.items():
    for id in ids:
        img_name = f'image_{id}.png'
        ann_name = f'annotation_{id}.png'
        
        shutil.copy(os.path.join(images_dir, img_name), os.path.join(images_dir, split, img_name))
        shutil.copy(os.path.join(annotations_dir, ann_name), os.path.join(annotations_dir, split, ann_name))

print("\n✅ Dataset split complete:")
print(f" - Train: {len(train_ids)}")
print(f" - Val:   {len(val_ids)}")
print(f" - Test:  {len(test_ids)}")

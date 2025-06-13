import os
import shutil

# Set your main dataset path
dataset_dir = './nelissen_dataset'  # <-- Change this

images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')

# Get sorted list of images and annotations
image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.png')])
annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.lower().endswith('.png')])


# Sanity check: same number of files
if len(image_files) != len(annotation_files):
    raise ValueError(f"Mismatch: {len(image_files)} images vs {len(annotation_files)} annotations")

# Rename and overwrite
for idx, (img_file, ann_file) in enumerate(zip(image_files, annotation_files)):
    new_img_name = f'image_{idx}.png'
    new_ann_name = f'annotation_{idx}.png'

    # Full paths
    old_img_path = os.path.join(images_dir, img_file)
    new_img_path = os.path.join(images_dir, new_img_name)

    old_ann_path = os.path.join(annotations_dir, ann_file)
    new_ann_path = os.path.join(annotations_dir, new_ann_name)

    # Rename (overwrite if names already exist)
    os.rename(old_img_path, new_img_path)
    os.rename(old_ann_path, new_ann_path)

print(f"Renamed {len(image_files)} images and annotations.")

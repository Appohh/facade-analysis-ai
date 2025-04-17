from datasets import load_dataset
import os
from PIL import Image
import io

Define base directory
base_dir = "./cmp_facade_dataset"
splits = ["train", "test", "eval"]

Create main folders
for folder in ["images", "annotations"]:
    for split in splits:
        os.makedirs(os.path.join(base_dir, folder, split), exist_ok=True)

Loop through each split and process the dataset
for split in splits:
    print(f"Processing split: {split}")
    dataset = load_dataset("Xpitfire/cmp_facade", split=split)

    for idx, example in enumerate(dataset):
        # Load image and annotation from bytes
        image = Image.open(io.BytesIO(example["pixel_values"]["bytes"]))
        annotation = Image.open(io.BytesIO(example["label"]["bytes"]))

        # Define paths
        image_path = os.path.join(basedir, "images", split, f"image{idx}.png")
        annotation_path = os.path.join(basedir, "annotations", split, f"annotation{idx}.png")

        # Save files
        image.save(image_path)
        annotation.save(annotation_path)

print(f"\nAll images and annotations have been downloaded and organized in '{base_dir}'")
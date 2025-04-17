from datasets import load_dataset
import os
from PIL import Image
import io

# Load the dataset
dataset = load_dataset("Xpitfire/cmp_facade", split="eval")

# Define a directory to save images and annotations
save_dir = "./cmp_facade_dataset"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)

# Save each image and annotation
for idx, example in enumerate(dataset):
    # Load image and annotation from bytes
    image = Image.open(io.BytesIO(example["pixel_values"]["bytes"]))
    annotation = Image.open(io.BytesIO(example["label"]["bytes"]))
    
    # Save as PNG files
    image.save(os.path.join(save_dir, "images_eval", f"image_{idx}.png"))
    annotation.save(os.path.join(save_dir, "annotations_eval", f"annotation_{idx}.png"))

print(f"Downloaded and saved {len(dataset)} images and annotations to {save_dir}")
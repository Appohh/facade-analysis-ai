import os
import numpy as np
from PIL import Image

# Base directory
base_dir = "./data/cmp_facade_dataset"

def extract_and_save_coordinates(output_dir):
    # Create output directory structure
    sets = ["train", "eval", "test"]
    for set_name in sets:
        os.makedirs(os.path.join(output_dir, set_name), exist_ok=True)

    # Process each set
    for set_name in sets:
        annot_dir = os.path.join(base_dir, "annotations", set_name)
        output_set_dir = os.path.join(output_dir, set_name)

        # Get sorted annotation files
        annot_files = sorted([f for f in os.listdir(annot_dir) if f.endswith(".png")])

        # Process each annotation file
        for annot_file in annot_files:
            annot_path = os.path.join(annot_dir, annot_file)
            output_file = os.path.join(output_set_dir, annot_file.replace(".png", ".txt"))

            # Load annotation
            with Image.open(annot_path) as annot:
                annot_np = np.array(annot)

            # Extract coordinates for each class (1 to 12)
            coords_per_class = {i: [] for i in range(1, 13)}  # Classes 1 to 12
            height, width = annot_np.shape
            for y in range(height):
                for x in range(width):
                    class_id = annot_np[y, x]
                    if 1 <= class_id <= 12:  # Limit to 1-12
                        coords_per_class[class_id].append((x, y))

            # Save coordinates to text file
            with open(output_file, "w") as f:
                for class_id, coords in coords_per_class.items():
                    if coords:
                        f.write(f"Class {class_id}:\n")
                        f.write(" ".join([f"({x},{y})" for x, y in coords]) + "\n")

    print(f"Coordinates extracted and saved to {output_dir}")

if __name__ == "__main__":
    output_dir = os.path.join(base_dir, "annotations_coords1")

    extract_and_save_coordinates(output_dir)
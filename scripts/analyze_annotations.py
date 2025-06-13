import os
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Main annotation directory
ANNOT_ROOT = Path("nelissen_dataset/annotations")

splits = ["train", "eval", "test"]
global_value_counts = Counter()
global_image_value_map = defaultdict(set)

print("📊 Per-split Annotation Class Analysis:\n")

for split in splits:
    split_dir = ANNOT_ROOT / split
    value_counts = Counter()
    image_value_map = defaultdict(set)

    if not split_dir.exists():
        print(f"❌ Split directory not found: {split_dir}")
        continue

    for annot_path in split_dir.glob("*.png"):
        try:
            mask = np.array(Image.open(annot_path).convert("L"))
            unique, counts = np.unique(mask, return_counts=True)
            value_counts.update(dict(zip(unique, counts)))

            for val in unique:
                image_value_map[val].add(annot_path.name)

            # Add to global
            global_value_counts.update(dict(zip(unique, counts)))
            for val in unique:
                global_image_value_map[val].add(annot_path.name)

        except Exception as e:
            print(f"❗ Error reading {annot_path.name}: {e}")

    # Print per-split results
    print(f"🔹 Split: {split}")
    print(" Value  |  Total Pixels  |  Appears in # Images")
    print("--------|----------------|----------------------")
    for val in sorted(value_counts.keys()):
        print(f"{val:>6}  |  {value_counts[val]:>13,}  |  {len(image_value_map[val]):>20}")
    print()

# Print global summary
print("\n📦 Global Summary Across All Splits:")
print(" Value  |  Total Pixels  |  Appears in # Images")
print("--------|----------------|----------------------")
for val in sorted(global_value_counts.keys()):
    print(f"{val:>6}  |  {global_value_counts[val]:>13,}  |  {len(global_image_value_map[val]):>20}")

print("\n✅ Analysis complete.")

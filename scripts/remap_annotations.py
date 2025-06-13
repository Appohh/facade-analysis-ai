import os
from pathlib import Path
import numpy as np
from PIL import Image

# Input and output directories
ANNOT_IN_ROOT = Path("nelissen_dataset/annotations")
ANNOT_OUT_ROOT = Path("nelissen_dataset/annotations_mapped")

# Define the remapping
VALUE_TO_INDEX = {
    19: 0,
    29: 1,
    78: 2,
    79: 3,
}

splits = ["train", "eval", "test"]
ANNOT_OUT_ROOT.mkdir(parents=True, exist_ok=True)

for split in splits:
    in_dir = ANNOT_IN_ROOT / split
    out_dir = ANNOT_OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 Processing split: {split}")

    for mask_path in in_dir.glob("*.png"):
        mask = np.array(Image.open(mask_path).convert("L"))

        # Find unexpected pixel values
        unique_vals = np.unique(mask)
        unknown_vals = [v for v in unique_vals if v not in VALUE_TO_INDEX]
        if unknown_vals:
            print(f"❗ {mask_path.name}: Unknown pixel values: {unknown_vals}")
            continue  # Skip invalid masks

        # Map values
        remapped = np.vectorize(VALUE_TO_INDEX.get)(mask).astype(np.uint8)

        # Save
        out_path = out_dir / mask_path.name
        Image.fromarray(remapped).save(out_path)

    print(f"✅ Done: {split}")

print("\n🏁 All annotation masks remapped and saved to:", ANNOT_OUT_ROOT)

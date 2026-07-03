import os
import csv
import random
import shutil

NOT_SHORELINE_TILE_LIMIT = 3041

random.seed(42)
base_dir = "dataset"

# Source folders (from your tiling step)
src = {
    "shoreline": {
        "images": os.path.join(base_dir, "shoreline/images"),
        "masks":  os.path.join(base_dir, "shoreline/masks"),
    },
    "not_shoreline": {
        "images": os.path.join(base_dir, "not_shoreline/images"),
        "masks":  os.path.join(base_dir, "not_shoreline/masks"),
    }
}

# Destination folders
dst = {
    "train": {
        "images": os.path.join(base_dir, "train/images"),
        "masks":  os.path.join(base_dir, "train/masks"),
    },
    "val": {
        "images": os.path.join(base_dir, "val/images"),
        "masks":  os.path.join(base_dir, "val/masks"),
    }
}

# Make destination directories
for split in dst:
    for t in dst[split]:
        os.makedirs(dst[split][t], exist_ok=True)

# Load labels
labels = {"shoreline": [], "not_shoreline": []}

with open(os.path.join(base_dir, "labels.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels[row["label"]].append(row["tile"])

# --- Subsample not_shoreline tiles ---
not_shoreline_subset = random.sample(labels["not_shoreline"], NOT_SHORELINE_TILE_LIMIT)
labels["not_shoreline"] = not_shoreline_subset

# Stratified split
for cls in ["shoreline", "not_shoreline"]:
    files = labels[cls]
    random.shuffle(files)

    n_train = int(0.7 * len(files))
    train_files = files[:n_train]
    val_files   = files[n_train:]

    for split, file_list in [("train", train_files), ("val", val_files)]:
        for fname in file_list:
            msk_fname = fname.replace("img", "mask", 1)  # replaces only first occurrence
            
            img_src = os.path.join(src[cls]["images"], fname)
            msk_src = os.path.join(src[cls]["masks"], msk_fname)
            
            # Safety check
            if not os.path.exists(img_src):
                print(f"WARNING: Missing image: {img_src}")
                continue
            if not os.path.exists(msk_src):
                print(f"WARNING: Missing mask: {msk_src}")
                continue

            shutil.move(img_src, os.path.join(dst[split]["images"], fname))
            shutil.move(msk_src, os.path.join(dst[split]["masks"], msk_fname))

print("Stratified train/val split complete.")

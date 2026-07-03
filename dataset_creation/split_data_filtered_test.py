import os
import csv
import random
import shutil

NOT_SHORELINE_TILE_LIMIT = 188

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
    "images": os.path.join(base_dir, "test/images"),
    "masks":  os.path.join(base_dir, "test/masks"),
}

# Make destination directories
os.makedirs(dst["images"], exist_ok=True)
os.makedirs(dst["masks"],  exist_ok=True)

# Load labels
labels = {"shoreline": [], "not_shoreline": []}

labels_path = os.path.join(base_dir, "labels.csv")
with open(labels_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        labels[row["label"]].append(row["tile"])
 
print(f"Shoreline tiles     : {len(labels['shoreline'])}")
print(f"Not-shoreline tiles : {len(labels['not_shoreline'])}")

# --- Subsample not_shoreline tiles ---
if NOT_SHORELINE_TILE_LIMIT is not None and len(labels["not_shoreline"]) > NOT_SHORELINE_TILE_LIMIT:
    labels["not_shoreline"] = random.sample(labels["not_shoreline"], NOT_SHORELINE_TILE_LIMIT)
    print(f"Not-shoreline tiles after cap: {len(labels['not_shoreline'])}")

# Move everything into test
moved   = 0
skipped = 0

for cls in ["shoreline", "not_shoreline"]:
    for fname in labels[cls]:
        msk_fname = fname.replace("img", "mask", 1)
 
        img_src = os.path.join(src[cls]["images"], fname)
        msk_src = os.path.join(src[cls]["masks"],  msk_fname)
 
        if not os.path.exists(img_src):
            print(f"WARNING: Missing image: {img_src}")
            skipped += 1
            continue
        if not os.path.exists(msk_src):
            print(f"WARNING: Missing mask: {msk_src}")
            skipped += 1
            continue
 
        shutil.move(img_src, os.path.join(dst["images"], fname))
        shutil.move(msk_src, os.path.join(dst["masks"],  msk_fname))
        moved += 1
 
print(f"\nDone. Moved {moved} tile pairs into {base_dir}/test/")
if skipped:
    print(f"Skipped {skipped} tiles due to missing files — check warnings above.")

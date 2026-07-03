"""
Parallel GeoTIFF Reprojection + Tiling Pipeline
================================================
Speeds up the original sequential script using:
  1. Multiprocessing — each image reprojected + tiled in a separate worker
  2. Compressed writes — DEFLATE on both reprojected files and tiles
  3. In-memory reprojection — avoids writing a temp file to disk at all
  4. Auto-deletes temp reprojected files after tiling

Usage:
    python prepare_dataset_test.py
    python prepare_dataset_test.py --workers 6
    python prepare_dataset_test.py --workers 6 --num_images 10

Requirements:
    pip install rasterio numpy tqdm
"""

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import os
import numpy as np
import csv
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TILE   = 512
STRIDE = 512

BASE_DIR    = "dataset"
IMAGES_ROOT = "shoreline_grids/deploy_test/images"
MASKS_ROOT  = "shoreline_grids/deploy_test/masks"

SHORE_IMG  = os.path.join(BASE_DIR, "shoreline/images")
SHORE_MSK  = os.path.join(BASE_DIR, "shoreline/masks")
NSHORE_IMG = os.path.join(BASE_DIR, "not_shoreline/images")
NSHORE_MSK = os.path.join(BASE_DIR, "not_shoreline/masks")


def reproject_to_memory(img_path, mask_path):
    """
    Reproject satellite image to match mask CRS/transform entirely in memory.
    Returns (reprojected_array, profile) without writing any temp file.
    """
    with rasterio.open(mask_path) as mask_ref:
        target_crs       = mask_ref.crs
        target_transform = mask_ref.transform
        target_width     = mask_ref.width
        target_height    = mask_ref.height

    with rasterio.open(img_path) as src:
        profile = src.profile.copy()
        profile.update({
            "crs"       : target_crs,
            "transform" : target_transform,
            "width"     : target_width,
            "height"    : target_height,
            "dtype"     : "float32",
            "compress"  : "deflate",
            "predictor" : 3,
            "zlevel"    : 9,
            "tiled"     : True,
            "blockxsize": 512,
            "blockysize": 512,
            "interleave": "band",
            "bigtiff"   : "IF_SAFER",
        })

        # Allocate full reprojected array in memory (bands, H, W)
        reprojected = np.zeros(
            (src.count, target_height, target_width), dtype=np.float32
        )

        for band in range(1, src.count + 1):
            dest = np.zeros((target_height, target_width), dtype=np.float32)
            reproject(
                source        = rasterio.band(src, band),
                destination   = dest,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = target_transform,
                dst_crs       = target_crs,
                resampling    = Resampling.bilinear
            )
            reprojected[band - 1] = dest

    return reprojected, profile, target_crs, target_transform


def process_one_image(args):
    """
    Worker function: reproject one image in memory, tile it, write tiles.
    Returns (image_idx, shoreline_count, not_shoreline_count, tile_records)
    where tile_records is a list of (fname_img, label).
    """
    i, img_path, mask_path = args

    # Safety check
    if not os.path.exists(img_path):
        return i, 0, 0, [], f"MISSING IMAGE: {img_path}"
    if not os.path.exists(mask_path):
        return i, 0, 0, [], f"MISSING MASK: {mask_path}"

    try:
        # ── Reproject in memory (no temp file written) ────────────────────
        reprojected, profile, target_crs, target_transform = \
            reproject_to_memory(img_path, mask_path)

        height, width = reprojected.shape[1], reprojected.shape[2]

        # Read mask
        with rasterio.open(mask_path) as mask_src:
            assert mask_src.width  == width,  "Width mismatch after reproject"
            assert mask_src.height == height, "Height mismatch after reproject"
            mask_full = mask_src.read(1)

        shoreline_count     = 0
        not_shoreline_count = 0
        tile_records        = []   # (fname_img, label, img_out_dir, msk_out_dir)

        tile_idx = 0
        for y in range(0, height - TILE + 1, STRIDE):
            for x in range(0, width - TILE + 1, STRIDE):

                img_tile  = reprojected[:, y:y+TILE, x:x+TILE]
                mask_tile = mask_full[y:y+TILE, x:x+TILE]

                has_shoreline = np.any(mask_tile > 0)

                if has_shoreline:
                    img_out = SHORE_IMG
                    msk_out = SHORE_MSK
                    label   = "shoreline"
                    shoreline_count += 1
                else:
                    img_out = NSHORE_IMG
                    msk_out = NSHORE_MSK
                    label   = "not_shoreline"
                    not_shoreline_count += 1

                tile_idx += 1
                idx       = f"{x}_{y}"
                fname_img = f"img{i+1}_{idx}_no{tile_idx}.tif"
                fname_msk = f"mask{i+1}_{idx}_no{tile_idx}.tif"

                win_transform = rasterio.transform.from_bounds(
                    target_transform.c + x * target_transform.a,
                    target_transform.f + (y + TILE) * target_transform.e,
                    target_transform.c + (x + TILE) * target_transform.a,
                    target_transform.f + y * target_transform.e,
                    TILE, TILE
                )

                # Replace NaN/Inf with 0 before casting
                img_tile = np.nan_to_num(img_tile, nan=0.0, posinf=65535.0, neginf=0.0)
                
                # Clip to valid uint16 range
                img_tile = np.clip(img_tile, 0, 65535)

                # ── Write image tile (compressed) ─────────────────────────
                with rasterio.open(
                    os.path.join(img_out, fname_img), "w",
                    driver    = "GTiff",
                    height    = TILE,
                    width     = TILE,
                    count     = reprojected.shape[0],
                    dtype     = "uint16",
                    crs       = target_crs,
                    transform = win_transform,
                    compress  = "deflate",
                    predictor = 2,
                    zlevel    = 9,
                ) as dst:
                    dst.write(img_tile.astype(np.uint16))

                # ── Write mask tile (compressed) ──────────────────────────
                with rasterio.open(
                    os.path.join(msk_out, fname_msk), "w",
                    driver    = "GTiff",
                    height    = TILE,
                    width     = TILE,
                    count     = 1,
                    dtype     = mask_tile.dtype,
                    crs       = target_crs,
                    transform = win_transform,
                    compress  = "deflate",
                    predictor = 2,
                    zlevel    = 9,
                ) as dst:
                    dst.write(mask_tile, 1)

                tile_records.append((fname_img, label))

        orig_mb = os.path.getsize(img_path) / 1e6
        return i, shoreline_count, not_shoreline_count, tile_records, \
               f"[OK] Image {i+1}: {orig_mb:.1f} MB | " \
               f"{shoreline_count} shoreline + {not_shoreline_count} non-shoreline tiles"

    except Exception as e:
        return i, 0, 0, [], f"[FAIL] Image {i+1}: {e}"


def main(num_images=110, num_workers=None):
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 6)

    # Create output directories
    for d in [SHORE_IMG, SHORE_MSK, NSHORE_IMG, NSHORE_MSK]:
        os.makedirs(d, exist_ok=True)

    # Build task list
    tasks = []
    for i in range(num_images):
        img_path  = os.path.join(IMAGES_ROOT, f"test_shoreline_grid_{i+1}_Bands.tif")
        mask_path = os.path.join(MASKS_ROOT,  f"test_shoreline_mask_{i+1}.tif")
        tasks.append((i, img_path, mask_path))

    print(f"\n{'='*60}")
    print(f"  Images to process : {num_images}")
    print(f"  Tile size         : {TILE}x{TILE}")
    print(f"  Stride            : {STRIDE}")
    print(f"  Workers           : {num_workers}")
    print(f"  Output directory  : {BASE_DIR}")
    print(f"{'='*60}\n")

    total_shoreline     = 0
    total_not_shoreline = 0
    total_tiles         = 0
    all_records         = []   # (fname_img, label) from all images
    failed              = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one_image, t): t for t in tasks}

        with tqdm(total=len(tasks), desc="Processing images", unit="img") as pbar:
            for future in as_completed(futures):
                i, s_count, ns_count, records, msg = future.result()

                if records is not None and len(records) > 0:
                    total_shoreline     += s_count
                    total_not_shoreline += ns_count
                    total_tiles         += s_count + ns_count
                    all_records.extend(records)
                    tqdm.write(f"  {msg}")
                else:
                    failed.append(msg)
                    tqdm.write(f"  {msg}")

                pbar.update(1)

    # ── Write labels.csv (after all workers finish) ───────────────────────────
    labels_path = os.path.join(BASE_DIR, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "label"])
        for fname, label in all_records:
            writer.writerow([fname, label])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Shoreline tiles     : {total_shoreline}")
    print(f"  Not-shoreline tiles : {total_not_shoreline}")
    print(f"  Total tiles         : {total_tiles}")
    print(f"  Labels CSV          : {labels_path}")
    if failed:
        print(f"\n  Failed images ({len(failed)}):")
        for msg in failed:
            print(f"    {msg}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel GeoTIFF reprojection + tiling pipeline."
    )
    parser.add_argument(
        "--workers", "-w", type=int,
        default=min(multiprocessing.cpu_count(), 6),
        help="Number of parallel workers (default: min(cpu_count, 6))"
    )
    parser.add_argument(
        "--num_images", "-n", type=int, default=10,
        help="Number of images to process (default: 10)"
    )
    args = parser.parse_args()
    main(num_images=args.num_images, num_workers=args.workers)
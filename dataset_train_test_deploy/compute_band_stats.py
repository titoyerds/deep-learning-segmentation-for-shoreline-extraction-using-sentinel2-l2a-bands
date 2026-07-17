"""
compute_band_stats.py
─────────────────────
Run this ONCE before training to compute per-band mean and std
from your training images.

Usage:
    python compute_band_stats.py

Output:
    data/band_stats.npz  ←  loaded automatically by prepare_data.py
"""

import os
import glob
import numpy as np
import rasterio
from tqdm import tqdm

SHORELINE_ROOT = "data/shoreline_dataset"
OUTPUT_PATH    = "data/band_stats.npz"

# Band names matching your GEE export order (for logging only)
BAND_NAMES = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']


def compute_stats(train_img_paths):
    """
    Welford's online algorithm — computes mean and variance in a single pass
    without loading all images into memory at once.

    Returns:
        mean : np.ndarray of shape (num_bands,)
        std  : np.ndarray of shape (num_bands,)
    """
    n_images = len(train_img_paths)
    assert n_images > 0, "No training images found!"

    # Read the first image to get band count and dtype
    with rasterio.open(train_img_paths[0]) as src:
        num_bands = src.count

    # Welford accumulators — one value per band
    count  = 0               # total pixels seen so far (same for all bands)
    mean   = np.zeros(num_bands, dtype=np.float64)
    M2     = np.zeros(num_bands, dtype=np.float64)   # sum of squared deviations

    print(f"\nComputing band statistics over {n_images} training images...")
    print(f"Number of bands detected: {num_bands}\n")

    for img_path in tqdm(train_img_paths, desc="Processing images"):
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float64)   # (bands, H, W)

        # Replace NaN/inf before accumulating
        np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Flatten spatial dims: (bands, H*W)
        pixels = image.reshape(num_bands, -1)   # (bands, H*W)
        n_new  = pixels.shape[1]                # number of pixels in this image

        # Welford update — vectorised over bands
        count   += n_new
        delta    = pixels - mean[:, np.newaxis]         # (bands, H*W)
        mean    += delta.sum(axis=1) / count            # running mean update
        delta2   = pixels - mean[:, np.newaxis]         # (bands, H*W) — updated mean
        M2      += (delta * delta2).sum(axis=1)         # running sum of sq. dev.

    variance = M2 / (count - 1)    # unbiased (sample) variance
    std      = np.sqrt(variance)

    return mean.astype(np.float32), std.astype(np.float32)


def main():
    train_img_dir = os.path.join(SHORELINE_ROOT, "train", "images")
    train_imgs    = sorted(glob.glob(os.path.join(train_img_dir, "img*.tif")))

    mean, std = compute_stats(train_imgs)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("BAND STATISTICS (computed from training set)")
    print("="*55)
    print(f"{'Band':<8} {'Mean':>10} {'Std':>10}")
    print("-"*30)
    for i, (b, m, s) in enumerate(zip(BAND_NAMES, mean, std)):
        print(f"{b:<8} {m:>10.2f} {s:>10.2f}")
    print("="*55)

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, mean=mean, std=std)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print("prepare_data.py will load these automatically on next run.")


if __name__ == "__main__":
    main()
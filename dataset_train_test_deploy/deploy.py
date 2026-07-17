import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import os
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt

from unet import UNet
from train_with_resume import ABLATION_CASES, NUM_CLASSES
from prepare_data import BAND_MEAN, BAND_STD, compute_index

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_RUNS_DIR  = "train_runs"
DEPLOY_RUNS_DIR = "deploy_runs"
TILE            = 512
STRIDE          = 512

INPUT_TIF = "assets/shoreline_image_palawan5.tif"

# ── Which ablation cases to deploy ──────────────────────────────────────────
# Set to None to deploy every case found in ABLATION_CASES (that has a
# trained checkpoint). Set to a list to deploy only those cases.
# Mirrors TEST_ONLY / RUN_ONLY in test.py and train_with_resume.py.
# Example: DEPLOY_ONLY = ["TC1_NaturalColor", "TC9_NDWI_Index"]
DEPLOY_ONLY = None

# ── Graph exclusion list ────────────────────────────────────────────────────
# Cases to omit from the latency-vs-score comparison graph only.
# Inference still runs and saves predictions/latency for these cases —
# they're just hidden from the plot. Mirrors GRAPH_EXCLUDE in test.py.
GRAPH_EXCLUDE = []

# Path to the test.py summary CSV used to pair latency with accuracy scores.
# Adjust to the actual test_runs/<timestamp> folder you want to compare against.
SUMMARY_CSV = "test_runs/2026-06-18_23-37/summary_all_cases.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Preprocessing — must mirror ShorelineDataset.__getitem__ exactly ──────────

def preprocess_tile(img, sel_bands, comp_fn):
    """
    img      : np.float32 array, shape (C_full, H, W) — all bands from rasterio
    sel_bands: list of 0-based band indices for this case
    comp_fn  : None | "ndvi" | "ndmi" | "ndwi" for this case

    Returns a float32 tensor ready for the model, matching the logic in
    prepare_data.py ShorelineDataset.__getitem__.
    Parameters are passed explicitly — never read from module-level globals,
    which would go stale across a loop over multiple cases.
    """
    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if comp_fn is None:
        # Raw band selection + per-band normalisation (from band_stats.npz)
        tile = img[sel_bands, :, :]                               # (C, H, W)
        mean = BAND_MEAN[sel_bands, np.newaxis, np.newaxis]
        std  = BAND_STD[sel_bands,  np.newaxis, np.newaxis]
        tile = (tile - mean) / (std + 1e-6)
    else:
        # Spectral-index mode: compute_index expects two 2-D arrays
        band_a = img[sel_bands[0]]                                # (H, W)
        band_b = img[sel_bands[1]]                                # (H, W)
        index  = compute_index(band_a, band_b)                    # (H, W), ~[-1, 1]
        tile   = np.clip(index, -1.0, 1.0)[np.newaxis, :, :]     # (1, H, W)

    return torch.from_numpy(tile).unsqueeze(0)   # (1, C, H, W)


# ── Inference ─────────────────────────────────────────────────────────────────

def get_tile_origins(total_size, tile_size, stride):
    """
    Returns a list of starting coordinates that fully cover [0, total_size),
    including a final tile anchored at (total_size - tile_size) so the last
    partial strip is never skipped — even if it overlaps the previous tile.
    """
    if total_size <= tile_size:
        return [0]

    origins = list(range(0, total_size - tile_size + 1, stride))

    last_full_origin = origins[-1] if origins else 0
    if last_full_origin + tile_size < total_size:
        origins.append(total_size - tile_size)

    return origins


def predict_geotiff(model, input_tif, output_mask_tif, n_bands, sel_bands, comp_fn):
    """
    model    : the loaded UNet for THIS case.
    n_bands  : number of input channels for the current case's model.
    sel_bands: band indices for this case (passed to preprocess_tile).
    comp_fn  : index mode for this case (passed to preprocess_tile).
    All are passed explicitly — never read from module-level globals,
    which would go stale when looping over multiple cases.
    """
    tile_times = []

    with rasterio.open(input_tif) as src:
        profile       = src.profile
        width, height = src.width, src.height

        out_mask    = np.zeros((height, width), dtype=np.uint8)
        confidence  = np.zeros((height, width), dtype=np.float32)
        total_tiles = shoreline_tiles = 0

        y_origins = get_tile_origins(height, TILE, STRIDE)
        x_origins = get_tile_origins(width,  TILE, STRIDE)

        # GPU warmup — use n_bands for THIS case
        dummy = torch.zeros(1, n_bands, TILE, TILE, device=device)
        with torch.no_grad():
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for y in y_origins:
            for x in x_origins:
                window = Window(x, y, TILE, TILE)
                img    = src.read(window=window).astype(np.float32)
                tensor = preprocess_tile(img, sel_bands, comp_fn).to(device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_start = time.perf_counter()

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred  = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                    prob1 = probs[0, 1].cpu().numpy()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                tile_times.append((t_end - t_start) * 1000)   # ms

                out_mask[y:y+TILE, x:x+TILE]   = pred
                confidence[y:y+TILE, x:x+TILE] = prob1
                total_tiles += 1
                if pred.max() > 0:
                    shoreline_tiles += 1

    tile_times = np.array(tile_times)
    latency_stats = {
        "mean_ms":     float(tile_times.mean()),
        "median_ms":   float(np.median(tile_times)),
        "p95_ms":      float(np.percentile(tile_times, 95)),
        "min_ms":      float(tile_times.min()),
        "max_ms":      float(tile_times.max()),
        "fps":         float(1000 / tile_times.mean()),
        "total_tiles": total_tiles,
    }

    print(f"\nTiles processed     : {total_tiles}")
    print(f"Tiles with shoreline: {shoreline_tiles}")
    print(f"Shoreline pixels    : {out_mask.sum()}")
    print(f"\nLatency (per tile):")
    print(f"  Mean   : {latency_stats['mean_ms']:.1f} ms")
    print(f"  Median : {latency_stats['median_ms']:.1f} ms")
    print(f"  P95    : {latency_stats['p95_ms']:.1f} ms")
    print(f"  FPS    : {latency_stats['fps']:.2f} tiles/sec")

    np.save(output_mask_tif.replace(".tif", "_tile_times.npy"), tile_times)

    profile.pop("nodata", None)
    profile.update(count=1, dtype="uint8", compress="deflate")

    with rasterio.open(output_mask_tif, "w", **profile) as dst:
        dst.write(out_mask, 1)

    conf_path = output_mask_tif.replace(".tif", "_confidence.tif")
    conf_profile = profile.copy()
    conf_profile.update(dtype="float32")
    with rasterio.open(conf_path, "w", **conf_profile) as dst:
        dst.write(confidence, 1)

    print(f"\nPrediction mask saved : {output_mask_tif}")
    print(f"Confidence map saved  : {conf_path}")

    return latency_stats


# ── Vectorisation ──────────────────────────────────────────────────────────────

def mask_to_vector(mask_tif, output_gpkg):
    with rasterio.open(mask_tif) as src:
        mask      = src.read(1)
        transform = src.transform
        crs       = src.crs

    shapes_gen = shapes(mask, mask=mask == 1, transform=transform)
    geoms      = [shape(geom) for geom, val in shapes_gen if val == 1]

    if not geoms:
        print("WARNING: No shoreline polygons found — check the confidence map in QGIS.")
        return

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_m2"] = gdf_proj.geometry.area
    gdf.to_file(output_gpkg, driver="GPKG")
    print(f"Vector shoreline saved: {output_gpkg}  ({len(geoms)} polygon(s))")


# ── Latency summary CSV ───────────────────────────────────────────────────────

def save_latency_summary_csv(latency_results, output_dir):
    """
    Writes one row per ablation case with its full latency_stats dict.
    Mirrors the structure of test.py's summary_all_cases.csv.
    """
    if not latency_results:
        print("No latency results to save.")
        return

    metrics = ["mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms",
               "fps", "total_tiles"]

    path = os.path.join(output_dir, "latency_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case"] + metrics)
        for case_name, stats in latency_results.items():
            row = [case_name] + [stats[m] for m in metrics]
            writer.writerow(row)

    print(f"\nLatency summary saved to: {path}")


# ── Graphs ─────────────────────────────────────────────────────────────────────

def plot_latency_vs_scores(latency_results, score_results, output_dir, exclude=None):
    """
    Scatter plot: x = mean latency (ms), y = each metric score.
    One point per ablation case, labeled with the case short name.
    exclude: case names to omit from this graph only.
             Defaults to the module-level GRAPH_EXCLUDE list.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    if exclude is None:
        exclude = GRAPH_EXCLUDE or []

    os.makedirs(output_dir, exist_ok=True)

    common = [c for c in latency_results if c in score_results and c not in exclude]
    if not common:
        print("\nNo cases left to plot after exclusions — skipping latency vs score graph.")
        return

    short_names = [c.split("_", 1)[1] if "_" in c else c for c in common]
    latencies   = [latency_results[c]["mean_ms"] for c in common]
    cmap        = plt.get_cmap("tab10")

    score_metrics = [
        ("mIoU",        "mIoU"),
        ("boundary_f1", "Boundary F1"),
        ("SCR",         "SCR"),
    ]

    fig = plt.figure(figsize=(22, 6))
    fig.suptitle("Inference Latency vs. Score per Ablation Case",
                 fontsize=13, fontweight="bold", y=1.01)
    gs   = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.38], figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_legend = fig.add_subplot(gs[0, 3])
    ax_legend.axis("off")

    for ax, (metric_key, metric_label) in zip(axes, score_metrics):
        scores = [score_results[c][metric_key] for c in common]

        for i, (lat, score) in enumerate(zip(latencies, scores)):
            ax.scatter(lat, score, color=cmap(i % 10), s=120, zorder=3,
                       edgecolors="white", linewidths=0.5)

        ax.set_xlabel("Mean Latency per Tile (ms)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Latency vs {metric_label}", fontsize=11)
        ax.grid(True, alpha=0.3)

        lat_arr   = np.array(latencies)
        score_arr = np.array(scores)
        lat_pad   = (lat_arr.max() - lat_arr.min()) * 0.25 + 1.0
        score_pad = (score_arr.max() - score_arr.min()) * 0.35 + 0.02
        ax.set_xlim(lat_arr.min() - lat_pad, lat_arr.max() + lat_pad)
        ax.set_ylim(score_arr.min() - score_pad, score_arr.max() + score_pad)

    legend_handles = [
        mpatches.Patch(color=cmap(i % 10), label=short)
        for i, short in enumerate(short_names)
    ]
    ax_legend.legend(
        handles=legend_handles,
        title="Ablation Cases",
        title_fontsize=10,
        fontsize=9,
        loc="center left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        borderpad=0.8,
        labelspacing=0.6,
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "graph_latency_vs_scores.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Per-case deploy ────────────────────────────────────────────────────────────

def deploy_case(case_name, case_cfg, run_dir):
    """
    Loads the model for one ablation case and runs inference on INPUT_TIF.
    Saves prediction mask, confidence map, and vector shoreline into a
    dedicated sub-folder for this case inside run_dir.
    Returns the latency_stats dict, or None if no checkpoint is found.
    """
    ckpt_path = os.path.join(TRAIN_RUNS_DIR, case_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] {case_name} — no checkpoint found at {ckpt_path}")
        return None

    selected_bands = case_cfg["indices"]
    compute_fn     = case_cfg["compute_fn"]
    num_bands      = 1 if compute_fn else len(selected_bands)

    model = UNet(num_classes=NUM_CLASSES, num_bands=num_bands).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\n{'='*60}")
    print(f"  Deploying: {case_name}")
    print(f"  Bands:     {selected_bands}  |  compute_fn: {compute_fn or 'raw'}")
    print(f"  Num bands (in): {num_bands}")
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"  Best val F1:    {checkpoint.get('val_f1', 'N/A')}")
    print(f"{'='*60}")

    case_dir = os.path.join(run_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    pred_mask = os.path.join(case_dir, f"predicted_mask_{case_name}.tif")
    gpkg      = os.path.join(case_dir, f"predicted_shoreline_{case_name}.gpkg")

    latency_stats = predict_geotiff(
        model, INPUT_TIF, pred_mask,
        n_bands=num_bands, sel_bands=selected_bands, comp_fn=compute_fn
    )
    mask_to_vector(pred_mask, gpkg)

    torch.cuda.empty_cache()
    return latency_stats


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {device}\n")

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir  = os.path.join(DEPLOY_RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Deploy run directory: {run_dir}\n")

    cases_to_run = list(ABLATION_CASES.keys()) if DEPLOY_ONLY is None else DEPLOY_ONLY
    invalid = [c for c in cases_to_run if c not in ABLATION_CASES]
    assert not invalid, (
        f"Unknown case(s) in DEPLOY_ONLY: {invalid}. "
        f"Choose from: {list(ABLATION_CASES.keys())}"
    )

    latency_results = {}

    for case_name in cases_to_run:
        stats = deploy_case(case_name, ABLATION_CASES[case_name], run_dir)
        if stats is not None:
            latency_results[case_name] = stats

    save_latency_summary_csv(latency_results, run_dir)

    score_results = {}
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline="") as f:
            for row in csv.DictReader(f):
                score_results[row["case"]] = {k: float(v) for k, v in row.items()
                                               if k != "case"}
    else:
        print(f"\n[WARNING] Summary CSV not found at: {SUMMARY_CSV}")
        print("           Run test.py first, or update SUMMARY_CSV at the top of this file.")

    if latency_results and score_results:
        if GRAPH_EXCLUDE:
            print(f"\nExcluding from graph: {GRAPH_EXCLUDE}")
        plot_latency_vs_scores(latency_results, score_results, run_dir)
    else:
        print("\nCould not generate latency vs score graph "
              "— run test.py first to produce summary_all_cases.csv.")

    print(f"\nAll outputs saved under: {run_dir}")
"""
Shoreline Prediction Demo App
=============================
Interactive Streamlit app for demonstrating the trained UNet shoreline
detection model. Upload a Sentinel-2 GeoTIFF, pick which ablation-case
model to use, and view/download the predicted shoreline mask and vector
overlay.

Run with:
    streamlit run demo_app.py
"""

import streamlit as st
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
import time

from unet import UNet
from train_with_resume import ABLATION_CASES, NUM_CLASSES
from prepare_data import BAND_MEAN, BAND_STD, compute_index

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_RUNS_DIR = "train_runs"
TILE           = 512
STRIDE         = 512
UPLOAD_CHUNK   = 8 * 1024 * 1024  # 8MB chunks when writing to disk
PREVIEW_MAXDIM = 1024             # downsample cap for the RGB preview

st.set_page_config(page_title="Shoreline Detection Demo", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Cache model loading so switching cases doesn't reload every rerun ────────
@st.cache_resource
def load_model(case_name):
    case_cfg   = ABLATION_CASES[case_name]
    sel_bands  = case_cfg["indices"]
    compute_fn = case_cfg["compute_fn"]
    num_bands  = 1 if compute_fn else len(sel_bands)

    ckpt_path = os.path.join(TRAIN_RUNS_DIR, case_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        return None, None, None, None, f"No checkpoint found at {ckpt_path}"

    model = UNet(num_classes=NUM_CLASSES, num_bands=num_bands).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_f1 = checkpoint.get("val_f1", None)
    return model, sel_bands, compute_fn, val_f1, None


def preprocess_tile(img, sel_bands, comp_fn):
    np.nan_to_num(img, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if comp_fn is None:
        tile = img[sel_bands, :, :]
        mean = BAND_MEAN[sel_bands, np.newaxis, np.newaxis]
        std  = BAND_STD[sel_bands,  np.newaxis, np.newaxis]
        tile = (tile - mean) / (std + 1e-6)
    else:
        band_a = img[sel_bands[0]]
        band_b = img[sel_bands[1]]
        index  = compute_index(band_a, band_b)
        tile   = np.clip(index, -1.0, 1.0)[np.newaxis, :, :]
    return torch.from_numpy(tile).unsqueeze(0)


def get_tile_origins(total_size, tile_size, stride):
    if total_size <= tile_size:
        return [0]
    origins = list(range(0, total_size - tile_size + 1, stride))
    if origins[-1] + tile_size < total_size:
        origins.append(total_size - tile_size)
    return origins


def run_inference(model, tif_path, sel_bands, comp_fn, num_bands, progress_bar):
    with rasterio.open(tif_path) as src:
        profile = src.profile
        width, height = src.width, src.height

        out_mask   = np.zeros((height, width), dtype=np.uint8)
        confidence = np.zeros((height, width), dtype=np.float32)

        y_origins = get_tile_origins(height, TILE, STRIDE)
        x_origins = get_tile_origins(width,  TILE, STRIDE)
        total     = len(y_origins) * len(x_origins)
        done      = 0

        t0 = time.time()
        for y in y_origins:
            for x in x_origins:
                window = Window(x, y, TILE, TILE)
                img    = src.read(window=window).astype(np.float32)
                tensor = preprocess_tile(img, sel_bands, comp_fn).to(device)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                        logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred  = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                    prob1 = probs[0, 1].cpu().numpy()

                out_mask[y:y+TILE, x:x+TILE]   = pred
                confidence[y:y+TILE, x:x+TILE] = prob1
                done += 1
                progress_bar.progress(done / total)

        elapsed = time.time() - t0

    return out_mask, confidence, profile, elapsed


def mask_to_vector(mask, transform, crs):
    shapes_gen = shapes(mask, mask=mask == 1, transform=transform)
    geoms = [shape(geom) for geom, val in shapes_gen if val == 1]
    if not geoms:
        return None
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf


def make_rgb_preview(tif_path, max_dim=PREVIEW_MAXDIM):
    """Build a stretched RGB preview (B4,B3,B2), downsampled for display speed."""
    with rasterio.open(tif_path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
        nodata = src.nodata
        r = src.read(4, out_shape=out_shape).astype(np.float32)
        g = src.read(3, out_shape=out_shape).astype(np.float32)
        b = src.read(2, out_shape=out_shape).astype(np.float32)

    def stretch(band):
        # Treat NaN and the raster's declared nodata value as missing data so
        # they don't poison the percentile calc (NaN in -> NaN out -> black image).
        band = band.copy()
        if nodata is not None:
            band[band == nodata] = np.nan
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            return np.zeros_like(band)
        p2, p98 = np.nanpercentile(band, (2, 98))
        band = np.nan_to_num(band, nan=p2)  # fill missing pixels with the low end
        band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
        return band

    rgb = np.dstack([stretch(r), stretch(g), stretch(b)])
    return rgb


def _stretch_band(band, nodata):
    band = band.copy()
    if nodata is not None:
        band[band == nodata] = np.nan
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros_like(band)
    p2, p98 = np.nanpercentile(band, (2, 98))
    band = np.nan_to_num(band, nan=p2)
    return np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)


def make_full_res_rgb(tif_path):
    """Read B4/B3/B2 at native full resolution (no out_shape downsampling)
    and stretch."""
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        r = src.read(4).astype(np.float32)
        g = src.read(3).astype(np.float32)
        b = src.read(2).astype(np.float32)
    return np.dstack([_stretch_band(r, nodata), _stretch_band(g, nodata), _stretch_band(b, nodata)])


def make_full_res_band(tif_path, band_index):
    """Read a single band (e.g. Band 8 / NIR) at native full resolution and
    stretch, same reason as make_full_res_rgb."""
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        band = src.read(band_index).astype(np.float32)
    return _stretch_band(band, nodata)


def resize_array_to_shape(arr, target_shape):
    """Resize a float array (values in [0,1]) to an exact (height, width),
    using bilinear interpolation."""
    target_h, target_w = target_shape
    arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr_u8)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def make_band_preview(tif_path, band_index, max_dim=PREVIEW_MAXDIM):
    """Build a stretched single-band grayscale preview (e.g. Band 8 / NIR),
    downsampled for display speed. Returns a 2D float array in [0,1]."""
    with rasterio.open(tif_path) as src:
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
        nodata = src.nodata
        band = src.read(band_index, out_shape=out_shape).astype(np.float32)

    band = band.copy()
    if nodata is not None:
        band[band == nodata] = np.nan
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        return np.zeros_like(band)
    p2, p98 = np.nanpercentile(band, (2, 98))
    band = np.nan_to_num(band, nan=p2)
    band = np.clip((band - p2) / (p98 - p2 + 1e-6), 0, 1)
    return band


def downsample_for_display(arr, max_dim=1400):
    """Shrink a 2D or 3D array for browser-side rendering only."""
    h, w = arr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return arr
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        img = img.resize((new_w, new_h), Image.NEAREST)
        return np.asarray(img)
    else:
        arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr_u8)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        return np.asarray(img).astype(np.float32) / 255.0


def show_zoomable_image(array, key, is_rgb=True, colorscale=None, height=500):
    """Render an array as a Google-Maps-style pan/scroll-zoomable image using Plotly."""
    import io, base64

    if is_rgb:
        img_u8 = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_u8)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG", optimize=True)
        data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        h, w = img_u8.shape[:2]
        fig = go.Figure(go.Image(source=data_uri, x0=0, y0=0, dx=1, dy=1))
        fig.update_xaxes(range=[0, w])
        fig.update_yaxes(range=[h, 0])
    else:
        fig = go.Figure(go.Heatmap(z=array, colorscale=colorscale or "gray", showscale=False))
        fig.update_yaxes(autorange="reversed")

    fig.update_xaxes(visible=False, constrain="domain")
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        dragmode="pan",
        uirevision=key,  # keep zoom/pan state stable across reruns
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
        key=key,
    )


def save_upload_to_tempfile(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        uploaded_file.seek(0)
        while True:
            chunk = uploaded_file.read(UPLOAD_CHUNK)
            if not chunk:
                break
            tmp.write(chunk)
        return tmp.name


def cleanup_session_tempfile():
    """Remove any tempfile from a previous upload, tracked via session_state."""
    old_path = st.session_state.get("tif_path")
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except OSError:
            pass
    st.session_state.tif_path = None
    st.session_state.rgb_preview = None
    st.session_state.nir_preview = None
    st.session_state.rgb_full = None
    st.session_state.nir_full = None
    st.session_state.uploaded_file_id = None


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Philippine Shoreline Prediction - Live Demo")
st.caption("Upload a Sentinel-2 L2A GeoTIFF and view the model's predicted shoreline.")

# session_state defaults
for key in ("tif_path", "rgb_preview", "nir_preview", "rgb_full", "nir_full", "uploaded_file_id"):
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("Settings")

    available_cases = [
        c for c in ABLATION_CASES
        if os.path.exists(os.path.join(TRAIN_RUNS_DIR, c, "best_model.pth"))
    ]
    if not available_cases:
        st.error(f"No trained checkpoints found under `{TRAIN_RUNS_DIR}/`.")
        st.stop()

    case_name = st.selectbox(
        "Select model (band combination)",
        available_cases,
        index=available_cases.index("TC4_Agriculture") if "TC4_Agriculture" in available_cases else 0,
    )

    threshold = st.slider(
        "Confidence threshold override (visual only)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Adjust to see how the predicted mask changes at different "
             "confidence cutoffs. Does not affect the saved GeoPackage, "
             "which always uses the model's argmax prediction."
    )

    uploaded_file = st.file_uploader("Upload Sentinel-2 GeoTIFF (.tif)", type=["tif", "tiff"])

    overlay_bg = st.radio(
        "Overlay background",
        ["RGB (natural color)", "Band 8 (NIR) grayscale"],
        help="Band 8 (NIR) often makes the water/land boundary sharper and "
             "easier to check the predicted shoreline against.",
    )

# ── Load selected model ───────────────────────────────────────────────────────
model, sel_bands, compute_fn, val_f1, err = load_model(case_name)

if err:
    st.error(err)
    st.stop()

num_bands = 1 if compute_fn else len(sel_bands)

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Model", case_name.split("_", 1)[1] if "_" in case_name else case_name)
col_info2.metric("Input bands", num_bands)
col_info3.metric("Val F1 (at save)", f"{val_f1:.4f}" if val_f1 else "N/A")

# ── Handle upload: write to disk once per file, cache path in session_state ──
if uploaded_file is not None:
    # file_id changes whenever a genuinely new/different file is uploaded
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.uploaded_file_id != file_id:
        cleanup_session_tempfile()
        with st.spinner("Saving upload to disk..."):
            st.session_state.tif_path = save_upload_to_tempfile(uploaded_file)
        st.session_state.uploaded_file_id = file_id

        st.divider()
        st.subheader("Input Image Preview (Natural Color)")
        try:
            with st.spinner("Building preview..."):
                st.session_state.rgb_preview = make_rgb_preview(st.session_state.tif_path)
        except Exception as e:
            st.warning(f"Could not build RGB preview: {e}")
            st.session_state.rgb_preview = None
    else:
        st.divider()
        st.subheader("Input Image Preview (Natural Color)")

    if st.session_state.rgb_preview is not None:
        st.image(st.session_state.rgb_preview, use_container_width=True)

    tif_path = st.session_state.tif_path
    rgb_preview = st.session_state.rgb_preview

    if st.button("Run Shoreline Prediction", type="primary"):
        progress_bar = st.progress(0.0)
        status = st.empty()
        status.info("Running inference tile by tile...")

        out_mask, confidence, profile, elapsed = run_inference(
            model, tif_path, sel_bands, compute_fn, num_bands, progress_bar
        )
        status.success(f"Inference complete in {elapsed:.1f} seconds "
                       f"({out_mask.shape[0]}×{out_mask.shape[1]} pixels).")

        # ── Build the overlay background at true full resolution ──────────────
        if overlay_bg.startswith("Band 8"):
            cache_key = "nir_full"
            if st.session_state.get(cache_key) is None:
                with st.spinner("Reading Band 8 (NIR) at full resolution..."):
                    band_full = make_full_res_band(tif_path, band_index=8)
                    st.session_state[cache_key] = np.dstack([band_full] * 3)
            bg_full = st.session_state[cache_key]
        else:
            cache_key = "rgb_full"
            if st.session_state.get(cache_key) is None:
                with st.spinner("Reading RGB at full resolution..."):
                    st.session_state[cache_key] = make_full_res_rgb(tif_path)
            bg_full = st.session_state[cache_key]

        # ── Display side-by-side ──────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Original**")
            if bg_full is not None:
                st.image(bg_full, use_container_width=True)

        with col2:
            st.markdown("**Predicted Shoreline Mask**")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(out_mask, cmap="Reds", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

        with col3:
            st.markdown(f"**Confidence Map (threshold={threshold:.2f})**")
            confidence_display = np.where(confidence >= threshold, confidence, 0.0)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(confidence_display, cmap="viridis", vmin=0, vmax=1)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

        # ── Overlay (the only zoomable/pannable panel) ────────────────────────
        st.markdown("**Overlay: Predicted Shoreline on Original Image**")
        if bg_full is not None:
            overlay_rgba = np.dstack([bg_full, np.ones(out_mask.shape, dtype=np.float32)])
            shoreline_color = np.array([1.0, 0.2, 0.2, 1.0])  # red, opaque
            alpha = 0.85
            mask_bool = out_mask == 1
            overlay_rgba[mask_bool] = (1 - alpha) * overlay_rgba[mask_bool] + alpha * shoreline_color
            overlay_display = downsample_for_display(overlay_rgba[..., :3])
            show_zoomable_image(overlay_display, key="result_overlay", is_rgb=True, height=650)
            st.caption("Scroll to zoom, click-drag to pan.")
        else:
            st.info("No background preview available to overlay on.")

        # ── Stats ──────────────────────────────────────────────────────────
        shoreline_px = int(out_mask.sum())
        total_px     = out_mask.size
        st.markdown(f"**Shoreline pixels detected:** {shoreline_px:,} "
                    f"({100 * shoreline_px / total_px:.3f}% of image)")

        # ── Downloads ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Downloads")

        out_dir = tempfile.mkdtemp()
        mask_path = os.path.join(out_dir, "predicted_mask.tif")
        gpkg_path = os.path.join(out_dir, "predicted_shoreline.gpkg")

        mask_profile = profile.copy()
        mask_profile.pop("nodata", None)
        mask_profile.update(count=1, dtype="uint8", compress="deflate")
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(out_mask, 1)

        gdf = mask_to_vector(out_mask, profile["transform"], profile["crs"])
        gpkg_available = gdf is not None
        if gpkg_available:
            gdf.to_file(gpkg_path, driver="GPKG")

        dl1, dl2 = st.columns(2)
        with dl1:
            with open(mask_path, "rb") as f:
                st.download_button("Download Predicted Mask (.tif)", f,
                                   file_name="predicted_mask.tif")
        with dl2:
            if gpkg_available:
                with open(gpkg_path, "rb") as f:
                    st.download_button("Download Shoreline Vector (.gpkg)", f,
                                       file_name="predicted_shoreline.gpkg")
            else:
                st.info("No shoreline polygons detected — GeoPackage not generated.")

else:
    # No file currently in the uploader widget - clean up any leftover tempfile
    if st.session_state.tif_path is not None:
        cleanup_session_tempfile()
    st.info("Upload a Sentinel-2 GeoTIFF in the sidebar to begin.")
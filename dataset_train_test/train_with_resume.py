import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from unet import UNet, DiceCELoss
from prepare_data import prepare_train_val_path, prepare_dataset
import time
from datetime import datetime
import sys
from tqdm.auto import tqdm

NUM_EPOCHS = 500
BATCH_SIZE = 16
LR = 0.0001
NUM_CLASSES = 2
NUM_WORKERS = 4        # set to 0 if you hit errors on Windows
PATIENCE    = 20

# Maps rasterio 0-based index => GEE band name for your specific export
BAND_NAMES = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']

SHORELINE_ROOT = "data/shoreline_dataset"
TRAIN_RUNS_DIR = "train_runs"

# ─── Google Drive output directory ────────────────────────────────────────────
# Example: "/content/drive/MyDrive/shoreline_results/train_runs"
# Leave as None to disable Drive syncing.
GDRIVE_SAVE_DIR = "/content/drive/MyDrive/shorelinedataset_train/train_runs"

# ─── Ablation Study Band Configs ───────────────────────────────────────────────
# Band index mapping (0-based, order stored in GeoTIFF: B1=0, B2=1, ..., B12=11)
# "indices" => passed as selected_bands to ShorelineDataset
# "compute_fn" => optional callable(image_array) for index bands; None = raw bands

ABLATION_CASES = {
    #  Name                      bands in GeoTIFF                                  rasterio 0-based idx         channels
    "TC1_NaturalColor":      {"indices": [3, 2, 1],                              "compute_fn": None},       # B4, B3, B2          = 3
    "TC2_ColorInfrared":     {"indices": [7, 3, 2],                              "compute_fn": None},       # B8, B4, B3          = 3
    "TC3_ShortWaveInfrared": {"indices": [11, 8, 3],                             "compute_fn": None},       # B12, B8A, B4        = 3
    "TC4_Agriculture":       {"indices": [10, 7, 1],                             "compute_fn": None},       # B11, B8, B2         = 3
    "TC5_Geology":           {"indices": [11, 10, 1],                            "compute_fn": None},       # B12, B11, B2        = 3
    "TC6_Bathymetric":       {"indices": [3, 2, 0],                              "compute_fn": None},       # B4, B3, B1          = 3
    "TC7_VegetationIndex":   {"indices": [7, 3],                                 "compute_fn": "ndvi"},     # (B8-B4)/(B8+B4)     = 1
    "TC8_MoistureIndex":     {"indices": [8, 10],                                "compute_fn": "ndmi"},     # (B8A-B11)/(B8A+B11) = 1
    "TC9_NDWI_Index":        {"indices": [2, 7],                                 "compute_fn": "ndwi"},     # (B3-B8)/(B3+B8)     = 1
    "TC10_FullSpectrum":     {"indices": [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], "compute_fn": None},       # B11 - B0            = 12
}


# LOGGING
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()



# HELPERS
def check_train_data(train_images, train_masks):
    assert len(train_images) == len(train_masks)
    print("Found images:", len(train_images))
    print(" Found masks:", len(train_masks))
    assert len(train_images) > 0, "No training images found!"


def prepare_train_runs(runs_dir):
    # Create runs directory
    os.makedirs(runs_dir, exist_ok=True)

    # Timestamp for this run
    name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(runs_dir, name)
    os.makedirs(path)

    print(f"\nTraining runs directory: {path}")
    return path


def config(model=None, lr=LR):
    """
    Set up device, loss, and (for training) optimizer + scheduler.
    criterion CE weights are moved to the correct device here in one place.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
 
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
    criterion = DiceCELoss(alpha=1.0, ce_weight=0.5)

    # Move CE class weights to device
    criterion.ce.weight = criterion.ce.weight.to(device)
 
    if model is not None:   # training path
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        return device, criterion, optimizer, scheduler
    else:                   # testing path
        return device, criterion
    


# METRICS
def pixel_accuracy(pred, target):
    return (pred == target).float().mean()

def f1_score_per_class(pred, target, num_classes):
    f1s = []
    for cls in range(num_classes):
        pred_i   = pred == cls
        target_i = target == cls
 
        tp = ( pred_i &  target_i).sum().float()
        fp = ( pred_i & ~target_i).sum().float()
        fn = (~pred_i &  target_i).sum().float()
 
        denom = 2 * tp + fp + fn
        if denom == 0:
            continue
        f1s.append((2 * tp) / denom)
    return torch.mean(torch.stack(f1s))

def f1_score_shoreline_only(pred, target):
    """F1 for class 1 (shoreline) only — more meaningful for imbalanced detection."""
    pred_i   = (pred == 1)
    target_i = (target == 1)

    tp = (pred_i &  target_i).sum().float()
    fp = (pred_i & ~target_i).sum().float()
    fn = (~pred_i & target_i).sum().float()

    denom = 2 * tp + fp + fn
    if denom == 0:
        return torch.tensor(0.0)
    return 2 * tp / denom


def train(model, loader, device, optimizer, criterion, scaler, epoch, num_epochs):
    model.train()
    total_loss = total_dice = total_ce = 0.0
 
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{num_epochs} [Train]",
                leave=False, ncols=110)
 
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
 
        optimizer.zero_grad(set_to_none=True)
 
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            outputs = model(images)
            loss, dice_loss, ce_loss = criterion(
                outputs, masks, return_components=True
            )
 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
 
        total_loss += loss.item()
        total_dice += dice_loss.item()
        total_ce   += ce_loss.item()
 
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{dice_loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}"
        )
 
    n = len(loader)
    return total_loss / n, total_dice / n, total_ce / n


@torch.no_grad()
def validate(model, loader, device, num_classes, criterion):
    model.eval()
    total_acc = total_f1_mean = total_f1_shoreline = 0.0
    total_loss = total_dice = total_ce = 0.0
    count = 0
 
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
 
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            outputs = model(images)
            loss, dice_loss, ce_loss = criterion(
                outputs, masks, return_components=True
            )
 
        preds = torch.argmax(outputs, dim=1)
 
        total_loss         += loss.item()
        total_dice         += dice_loss.item()
        total_ce           += ce_loss.item()
        total_acc          += pixel_accuracy(preds, masks).item()
        total_f1_mean      += f1_score_per_class(preds, masks, num_classes).item()
        total_f1_shoreline += f1_score_shoreline_only(preds, masks).item()
        count      += 1
 
    n = max(count, 1)
    return total_f1_shoreline/n, total_f1_mean/n, total_acc/n, total_loss/n, total_dice/n, total_ce/n



# GOOGLE DRIVE SYNC
def _sync_to_drive(run_dir):
    """
    Copy the finished run folder to Google Drive.
    Works in Colab after: from google.colab import drive; drive.mount('/content/drive')
    Set GDRIVE_SAVE_DIR at the top of this file (or leave as None to skip).
    """
    if GDRIVE_SAVE_DIR is None:
        print("[Drive] GDRIVE_SAVE_DIR not set — skipping Drive sync.")
        return
    import shutil
    dest = os.path.join(GDRIVE_SAVE_DIR, os.path.basename(run_dir))
    try:
        shutil.copytree(run_dir, dest, dirs_exist_ok=True)
        print(f"[Drive] Run saved to Google Drive: {dest}")
    except Exception as e:
        print(f"[Drive] WARNING: Could not save to Drive — {e}")



# MAIN TRAINING LOOP
def train_model(train_loader, val_loader, num_epochs, num_classes, num_bands,
                lr, train_runs_dir, resume=False):

    model  = UNet(num_classes=num_classes, num_bands=num_bands)
    device, criterion, optimizer, scheduler = config(model, lr)
    model.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1           = 0.0
    epochs_no_improve = 0
    start_epoch       = 0      

    # ── Resume from checkpoint if requested ───────────────────────────────
    checkpoint_path = os.path.join(train_runs_dir, "best_model.pth")

    if resume and os.path.exists(checkpoint_path):
        print(f"\n[RESUME] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_f1           = ckpt["val_f1"]
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)   # backwards-compatible
        start_epoch       = ckpt["epoch"] + 1

        print(f"[RESUME] Resuming from epoch {start_epoch}  |  Best F1 so far: {best_f1:.4f}")
        print(f"[RESUME] epochs_no_improve restored to: {epochs_no_improve}\n")
    else:
        if resume:
            print(f"[RESUME] No checkpoint found at {checkpoint_path} — starting fresh.\n")

        # Fresh run: write config and create metrics CSV header
        with open(os.path.join(train_runs_dir, "config.txt"), "w") as f:
            f.write(f"epochs:        {NUM_EPOCHS}\n")
            f.write(f"batch_size:    {BATCH_SIZE}\n")
            f.write(f"learning_rate: {LR}\n")
            f.write(f"optimizer:     {optimizer.__class__.__name__}\n")
            f.write(f"scheduler:     ReduceLROnPlateau (mode=max, factor=0.5, patience=5, min_lr=1e-6)\n")
            f.write(f"loss:          Dice + CrossEntropy (DiceCELoss)\n")
            f.write(f"encoder:       ResNet-34 (pretrained)\n")
            f.write(f"amp:           {'enabled' if device.type == 'cuda' else 'disabled'}\n")
            f.write(f"early_stopping_patience: {PATIENCE}\n")

        metrics_path = os.path.join(train_runs_dir, "metrics.csv")
        with open(metrics_path, "w") as f:
            f.write("epoch,lr,"
                    "train_loss,train_dice,train_ce,"
                    "val_loss,val_dice,val_ce,"
                    "val_acc,val_f1_mean,val_f1_shoreline\n")

    metrics_path = os.path.join(train_runs_dir, "metrics.csv")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):    # <= starts at 0 or resumed epoch
        start_time = time.time()

        train_loss, train_dice, train_ce = \
            train(model, train_loader, device, optimizer, criterion, scaler, epoch, num_epochs)

        val_f1_shoreline, val_f1_mean, val_acc, val_loss, val_dice, val_ce = \
            validate(model, val_loader, device, num_classes, criterion)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1_shoreline)

        elapsed_time = time.time() - start_time
        mins, secs = int(elapsed_time // 60), int(elapsed_time % 60)

        # Append — safe for both fresh and resumed runs
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{current_lr:.6f},"
                    f"{train_loss:.6f},{train_dice:.6f},{train_ce:.6f},"
                    f"{val_loss:.6f},{val_dice:.6f},{val_ce:.6f},"
                    f"{val_acc:.6f},{val_f1_mean:.6f},{val_f1_shoreline:.6f}\n")

        print(
            f"Epoch {epoch:02d}: "
            f"Train Loss={train_loss:.4f} (Dice={train_dice:.4f}, CE={train_ce:.4f}) | "
            f"Val Loss={val_loss:.4f} (Dice={val_dice:.4f}, CE={val_ce:.4f}) | "
            f"Val Acc={val_acc:.4f} | "
            f"Val F1 Mean={val_f1_mean:.4f} | Val F1 Shoreline={val_f1_shoreline:.4f} | "
            f"Training Time={mins:02d}:{secs:02d}"
        )

        if val_f1_mean > best_f1:
            best_f1           = val_f1_mean
            epochs_no_improve = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_f1":               best_f1,
                "epochs_no_improve":    epochs_no_improve,   # <= save this too
            }, checkpoint_path)
            print(f"    Best model saved (Val F1 = {best_f1:.4f})\n")
            _sync_to_drive(train_runs_dir)
        else:
            epochs_no_improve += 1
            print(f"PATIENCE: {epochs_no_improve} / {PATIENCE}\n")
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} — no improvement for {PATIENCE} epochs.")
                break

    _sync_to_drive(train_runs_dir)
        

def verify_band_loading(loader, case_name, compute_fn, selected_bands):
    images, masks = next(iter(loader))
    print(f"\n[VERIFY] {case_name}")
    print(f"  Image shape : {images.shape}")
    print(f"  Mask unique : {torch.unique(masks).tolist()}")

    if compute_fn:
        a, b = selected_bands
        print(f"  Index formula : ({BAND_NAMES[a]} - {BAND_NAMES[b]}) / "
              f"({BAND_NAMES[a]} + {BAND_NAMES[b]})  [{compute_fn.upper()}]")
        print(f"  Value range   : [{images.min():.3f}, {images.max():.3f}]  "
              f"<= should be within [-1, 1]")
    else:
        gee_names = [BAND_NAMES[i] for i in selected_bands]
        print(f"  Band indices  : {selected_bands} => GEE names: {gee_names}")
        for c, name in enumerate(gee_names):
            ch = images[:, c, :, :]
            print(f"  Channel {c} [{name}]: "
                  f"min={ch.min():.3f}, max={ch.max():.3f}, "
                  f"mean={ch.mean():.3f}, std={ch.std():.3f}")
    print()


if __name__ == "__main__":
    RUN_ONLY = ["TC1_NaturalColor"]     # Set to None to run all

    _real_stdout = sys.stdout   # save originals once
    _real_stderr = sys.stderr

    print("Model Training Started!\n")

    train_imgs, train_masks, val_imgs, val_masks, _, _ = \
        prepare_train_val_path(root_dir=SHORELINE_ROOT)
    check_train_data(train_imgs, train_masks)

    for case_name, case_cfg in ABLATION_CASES.items():
        if RUN_ONLY is not None and case_name not in RUN_ONLY:
            print(f"Skipping {case_name}")
            continue

        selected_bands = case_cfg["indices"]
        compute_fn     = case_cfg["compute_fn"]
        num_bands      = 1 if compute_fn else len(selected_bands)

        run_dir = os.path.join(TRAIN_RUNS_DIR, case_name)
        os.makedirs(run_dir, exist_ok=True)

        # ── Auto-detect resume ────────────────────────────────────────────────
        checkpoint_exists = os.path.exists(os.path.join(run_dir, "best_model.pth"))
        resume            = checkpoint_exists
        if resume:
            print(f"\n[RESUME] Checkpoint found for {case_name} — resuming.")
        # ─────────────────────────────────────────────────────────────────────

        log_file = open(os.path.join(run_dir, "train.log"), "a" if resume else "w")  # append on resume
        sys.stdout = Tee(_real_stdout, log_file)
        sys.stderr = Tee(_real_stderr, log_file)

        try:
            train_loader, val_loader = prepare_dataset(
                batch_size=BATCH_SIZE,
                train_imgs=train_imgs, train_masks=train_masks,
                val_imgs=val_imgs,     val_masks=val_masks,
                selected_bands=selected_bands,
                compute_fn=compute_fn,
                num_workers=NUM_WORKERS
            )

            verify_band_loading(train_loader, case_name, compute_fn, selected_bands)

            train_model(
                train_loader=train_loader, val_loader=val_loader,
                num_epochs=NUM_EPOCHS,     num_classes=NUM_CLASSES,
                num_bands=num_bands,       lr=LR,
                train_runs_dir=run_dir,
                resume=resume
            )
        finally:
            if 'train_loader' in locals():
                del train_loader, val_loader
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log_file.close()
            sys.stdout = _real_stdout
            sys.stderr = _real_stderr
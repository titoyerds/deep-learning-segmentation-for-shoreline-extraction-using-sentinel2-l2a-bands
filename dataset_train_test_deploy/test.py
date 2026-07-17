import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for Colab and headless servers
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import binary_dilation
from prepare_data import prepare_train_val_path, make_test_loader
from train_with_resume import config, ABLATION_CASES, BATCH_SIZE, NUM_CLASSES, NUM_WORKERS, TRAIN_RUNS_DIR
from unet import UNet


# ── PATHS ──────────────────────────────────────────────────────────────────────────
GDRIVE_PROJECT_DIR        = "shorelinedataset_train"   # folder name inside My Drive

_GDRIVE_ROOT              = f"/content/drive/MyDrive/{GDRIVE_PROJECT_DIR}"

SHORELINE_ROOT            = "data/shoreline_dataset"          # local disk
_EFFECTIVE_TRAIN_RUNS_DIR = "train_runs"                      # local disk
TEST_RUNS_DIR             = f"{_GDRIVE_ROOT}/test_runs"       # Google Drive

# Set to None to test all cases, or list specific names to test only those.
# ex. TEST_ONLY = ["TC2_ColorInfrared", "TC3_ShortWaveInfrared", "TC4_Agriculture", "TC5_Geology",
#                  "TC6_Bathymetric", "TC7_VegetationIndex", "TC8_MoistureIndex", "TC9_NDWI_Index"]
TEST_ONLY = None

# Cases to exclude from all-case comparison graphs and the F1 overlay.
# These cases are still tested and their per-case CSVs/charts are saved normally.
GRAPH_EXCLUDE = []

# ── REGENERATE GRAPHS FROM PREVIOUS RUN ───────────────────────────────────────
# Set this to the path of a summary_all_cases.csv from a previous test run to
# regenerate the F1 overlay WITHOUT re-running any model inference.
# GRAPH_EXCLUDE is respected here too.
# Example:
#   PREVIOUS_SUMMARY_CSV = f"{_GDRIVE_ROOT}/test_runs/2026-06-17_04-59/summary_all_cases.csv"
# Set to None to skip (normal testing mode).
PREVIOUS_SUMMARY_CSV = "test_runs/2026-06-18_23-37/summary_all_cases.csv"


# ── SETUP ─────────────────────────────────────────────────────────────────────
def prepare_test_runs(runs_dir):
    os.makedirs(runs_dir, exist_ok=True)
    name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(runs_dir, name)
    os.makedirs(path)
    print(f"\nTest runs directory: {path}")
    return path


# ── CONFUSION MATRIX COMPONENTS ───────────────────────────────────────────────
def get_confusion_components(pred, target, num_classes):
    results = []
    for cls in range(num_classes):
        pred_i   = (pred == cls)
        target_i = (target == cls)
        tp = ( pred_i &  target_i).sum().float()
        tn = (~pred_i & ~target_i).sum().float()
        fp = ( pred_i & ~target_i).sum().float()
        fn = (~pred_i &  target_i).sum().float()
        results.append((tp, tn, fp, fn))
    return results


# ── METRICS ───────────────────────────────────────────────────────────────────
def get_accuracy(pred, target):
    """
    Overall (global) pixel accuracy = correctly classified pixels / total pixels.
    For class-level insight, see get_precision() and get_recall() per class.
    """
    return (pred == target).float().mean()

def get_precision(pred, target, num_classes):
    """Precision = TP / (TP + FP), averaged across classes that have predictions."""
    components = get_confusion_components(pred, target, num_classes)
    precisions = []
    for tp, tn, fp, fn in components:
        denom = tp + fp
        if denom == 0:
            continue
        precisions.append(tp / denom)
    return torch.mean(torch.stack(precisions))

def get_recall(pred, target, num_classes):
    """Recall = TP / (TP + FN), averaged across classes that have ground-truth pixels."""
    components = get_confusion_components(pred, target, num_classes)
    recalls = []
    for tp, tn, fp, fn in components:
        denom = tp + fn
        if denom == 0:
            continue
        recalls.append(tp / denom)
    return torch.mean(torch.stack(recalls))

def _extract_boundary(binary_mask, tolerance):
    """Dilate a binary mask and XOR with original to get boundary pixels."""
    dilated  = binary_dilation(binary_mask, iterations=tolerance)
    boundary = np.logical_xor(dilated, binary_mask)
    return boundary

def get_boundary_f1_score(pred, target, num_classes, tolerance=2):
    """BFScore = 2 * (Precision * Recall) / (Precision + Recall)"""
    pred_np   = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    bf_scores = []
    for cls in range(num_classes):
        pred_boundary   = _extract_boundary(pred_np   == cls, tolerance)
        target_boundary = _extract_boundary(target_np == cls, tolerance)

        tp = float(np.logical_and( pred_boundary,  target_boundary).sum())
        fp = float(np.logical_and( pred_boundary, ~target_boundary).sum())
        fn = float(np.logical_and(~pred_boundary,  target_boundary).sum())

        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        denom     = precision + recall
        if denom == 0:
            continue
        bf_scores.append(2 * precision * recall / denom)

    return float(np.mean(bf_scores)) if bf_scores else 0.0

def get_mIoU(pred, target, num_classes):
    components = get_confusion_components(pred, target, num_classes)
    ious = []
    for tp, tn, fp, fn in components:
        denom = tp + fp + fn
        if denom == 0:
            continue
        ious.append(tp / denom)
    return torch.mean(torch.stack(ious))

def _extract_shoreline(binary_mask):
    """Extract boundary pixels between water and land."""
    dilated = binary_dilation(binary_mask, iterations=1)
    return np.logical_xor(dilated, binary_mask)

def get_SCR(pred, target, tolerance=0):
    """
    SCR = 2 * |S1 ∩ S(S2, N)| / (|S1| + |S2|)

    S1 = ground-truth shoreline edge pixels
    S2 = predicted shoreline edge pixels
    tolerance=0  -> strict pixel overlap (paper's base SCR, Eq. 7)
    tolerance>0  -> S2 is dilated by N iterations (paper's SCR-N, Eq. 9)
    If both S1 and S2 are empty, returns 1.0 per paper's convention.
    """
    pred_np   = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    s1 = _extract_shoreline(target_np == 1)  # GT shoreline edges
    s2 = _extract_shoreline(pred_np   == 1)  # Pred shoreline edges

    if s1.sum() == 0 and s2.sum() == 0:
        return 1.0  # paper assigns SCR=1 when both are null

    if tolerance > 0:
        s2_expanded = binary_dilation(s2, iterations=tolerance)
    else:
        s2_expanded = s2

    intersection = np.logical_and(s1, s2_expanded).sum()
    denom = s1.sum() + s2.sum()

    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


# ── EVALUATE ONE CASE ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, test_loader, device, criterion, num_classes, test_runs_dir):
    model.eval()

    total_loss = total_dice_loss = total_ce_loss = 0.0
    total_acc = total_precision = total_recall = 0.0
    total_miou = total_bf1_sum = total_scr_sum = 0.0
    count = 0

    # ── NEW: accumulate for PR/ROC curves ─────────────────────────────────
    all_probs  = []   # shoreline class probability, flattened per image
    all_labels = []   # ground truth binary mask, flattened per image
    # ──────────────────────────────────────────────────────────────────────

    for images, masks in test_loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
            outputs = model(images)
            loss, dice_loss, ce_loss = criterion(outputs, masks, return_components=True)

        preds     = torch.argmax(outputs, dim=1)
        probs     = torch.softmax(outputs, dim=1)
        prob_shore = probs[:, 1, :, :]   # (B, H, W) — shoreline probability

        # ── NEW: collect flattened arrays ──────────────────────────────────
        all_probs.append(prob_shore.cpu().float().numpy().ravel())
        all_labels.append(masks.cpu().numpy().ravel())
        # ──────────────────────────────────────────────────────────────────

        total_loss      += loss.item()
        total_dice_loss += dice_loss.item()
        total_ce_loss   += ce_loss.item()
        total_acc       += get_accuracy(preds, masks).item()
        total_precision += get_precision(preds, masks, num_classes).item()
        total_recall    += get_recall(preds, masks, num_classes).item()
        total_miou      += get_mIoU(preds, masks, num_classes).item()
        total_bf1_sum   += get_boundary_f1_score(preds.cpu(), masks.cpu(), num_classes)
        total_scr_sum   += get_SCR(preds.cpu(), masks.cpu())
        count           += 1

    # ── Concatenate all collected arrays ──────────────────────────────────
    all_probs  = np.concatenate(all_probs)    # (N_total_pixels,)
    all_labels = np.concatenate(all_labels)   # (N_total_pixels,)

    total_bf1 = total_bf1_sum / max(count, 1)
    total_scr = total_scr_sum / max(count, 1)

    n = max(count, 1)
    results = {
        "test_loss":   total_loss      / n,
        "dice_loss":   total_dice_loss / n,
        "ce_loss":     total_ce_loss   / n,
        "accuracy":    total_acc       / n,
        "precision":   total_precision / n,
        "recall":      total_recall    / n,
        "boundary_f1": total_bf1,        # already a single dataset-level score
        "mIoU":        total_miou      / n,
        "SCR":         total_scr,        # already a single dataset-level score
    }

    print("\n" + "="*55)
    print("TEST RESULTS")
    print("="*55)
    print(f"  Total Loss        : {results['test_loss']:.4f}")
    print(f"    Dice Loss       : {results['dice_loss']:.4f}")
    print(f"    CE Loss         : {results['ce_loss']:.4f}")
    print(f"  Accuracy          : {results['accuracy']:.4f}")
    print(f"  Precision         : {results['precision']:.4f}")
    print(f"  Recall            : {results['recall']:.4f}")
    print(f"  Boundary F1-Score : {results['boundary_f1']:.4f}")
    print(f"  mIoU              : {results['mIoU']:.4f}")
    print(f"  SCR               : {results['SCR']:.4f}")
    print("="*55)

    # Save per-case CSV
    metrics_path = os.path.join(test_runs_dir, "test_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v:.6f}\n")
    print(f"\nTest metrics saved to: {metrics_path}")

    plot_pr_roc_curves(all_probs, all_labels, test_runs_dir)

    return results


# ── GRAPHS ────────────────────────────────────────────────────────────────────

def plot_training_curves(case_names, test_runs_dir, exclude=None):
    """
    Reads metrics.csv from each case's train_runs folder and plots:
      1. Loss curves      — train_loss vs val_loss per case (one PNG per case)
      2. F1 curves        — val_f1_mean per case (one PNG per case)
      3. LR schedule      — learning rate over epochs per case (one PNG per case)
      4. All-case overlay — val_f1_mean for all cases on one chart (comparison)
    Cases with no metrics.csv are silently skipped.
    Schema detection handles both old (val_f1 + sobel) and new (val_f1_mean + ce) formats.

    exclude: cases to omit from the all-case overlay only. Per-case charts
             (loss, F1, LR) are still generated for excluded cases.
             Defaults to the module-level GRAPH_EXCLUDE list.
    """
    import csv

    if exclude is None:
        exclude = GRAPH_EXCLUDE or []

    curves_dir = os.path.join(test_runs_dir, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)

    cmap        = plt.get_cmap("tab10")
    all_epochs  = {}    # case_name -> epoch list  (for overlay, excluded cases omitted)
    all_f1_mean = {}    # case_name -> val_f1_mean list

    for i, case_name in enumerate(case_names):
        csv_path = os.path.join(_EFFECTIVE_TRAIN_RUNS_DIR, case_name, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP training curves] No metrics.csv for {case_name}")
            continue

        with open(csv_path, newline="") as f:
            reader     = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            has_ce_schema    = "train_ce"    in fieldnames and "val_ce"    in fieldnames
            has_sobel_schema = "train_sobel" in fieldnames and "val_sobel" in fieldnames
            has_split_f1     = "val_f1_mean" in fieldnames

            if not has_ce_schema and not has_sobel_schema:
                print(f"  [SKIP training curves] Unrecognized CSV schema for "
                      f"{case_name}: columns={fieldnames}")
                continue

            third_key   = "train_ce" if has_ce_schema else "train_sobel"
            third_val   = "val_ce"   if has_ce_schema else "val_sobel"
            third_label = "CE Loss"  if has_ce_schema else "Sobel Loss"

            epochs      = []
            train_loss  = []
            val_loss    = []
            train_dice  = []
            val_dice    = []
            train_third = []
            val_third   = []
            f1_mean     = []
            lr_values   = []

            for row in reader:
                epochs.append(int(row["epoch"]))
                train_loss.append(float(row["train_loss"]))
                val_loss.append(float(row["val_loss"]))
                train_dice.append(float(row["train_dice"]))
                val_dice.append(float(row["val_dice"]))
                train_third.append(float(row[third_key]))
                val_third.append(float(row[third_val]))
                lr_values.append(float(row["lr"]))
                f1_mean.append(float(row["val_f1_mean"] if has_split_f1 else row["val_f1"]))

        short = case_name.split("_", 1)[1] if "_" in case_name else case_name

        # Only add to overlay if not excluded
        if case_name not in exclude:
            all_epochs[case_name]  = epochs
            all_f1_mean[case_name] = f1_mean

        # ── Chart 1: Loss curves (generated for ALL cases, even excluded) ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{short} — Loss Curves", fontsize=13, fontweight="bold")

        for ax, tr, vl, title in zip(
            axes,
            [train_loss, train_dice, train_third],
            [val_loss,   val_dice,   val_third],
            ["Total Loss", "Dice Loss", third_label]
        ):
            ax.plot(epochs, tr, label="Train", color="#4C72B0", linewidth=1.5)
            ax.plot(epochs, vl, label="Val",   color="#C44E52", linewidth=1.5, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            best_ep = epochs[int(np.argmin(vl))]
            ax.axvline(x=best_ep, color="gray", linestyle=":", linewidth=1.0,
                       label=f"Best epoch ({best_ep})")

        plt.tight_layout()
        path = os.path.join(curves_dir, f"{case_name}_loss_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── Chart 2: Val F1 Mean (generated for ALL cases, even excluded) ──
        fig, ax = plt.subplots(figsize=(10, 5))
        f1_label = "Val F1 Mean" if has_split_f1 else "Val F1"
        fig.suptitle(f"{short} — {f1_label}", fontsize=13, fontweight="bold")

        ax.plot(epochs, f1_mean, label=f1_label, color="#8172B2", linewidth=2)

        best_idx = int(np.argmax(f1_mean))
        best_ep  = epochs[best_idx]
        best_f1  = f1_mean[best_idx]
        ax.axvline(x=best_ep, color="gray", linestyle=":", linewidth=1.0)
        ax.annotate(f"Best: {best_f1:.4f}\n(epoch {best_ep})",
                    xy=(best_ep, best_f1),
                    xytext=(best_ep + max(1, len(epochs) * 0.03), best_f1 - 0.05),
                    fontsize=8, color="gray",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(curves_dir, f"{case_name}_f1_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── Chart 3: LR schedule (generated for ALL cases, even excluded) ──
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.suptitle(f"{short} — Learning Rate Schedule", fontsize=13, fontweight="bold")

        ax.plot(epochs, lr_values, color="#55A868", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")

        prev_lr = lr_values[0]
        for ep, lr in zip(epochs, lr_values):
            if lr < prev_lr:
                ax.axvline(x=ep, color="#C44E52", linestyle="--", linewidth=0.8, alpha=0.6)
                prev_lr = lr

        plt.tight_layout()
        path = os.path.join(curves_dir, f"{case_name}_lr_schedule.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        excluded_note = "  [excluded from overlay]" if case_name in exclude else ""
        print(f"  Training curves saved: {case_name}{excluded_note}")

    # ── Chart 4: All-case overlay — excluded cases not shown ──────────────
    if len(all_f1_mean) > 1:
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Rectangle

        all_final_f1 = [f[-1] for f in all_f1_mean.values()]
        y_min = min(all_final_f1) - 0.02
        y_max = max(all_final_f1) + 0.015
        max_ep = max(ep[-1] for ep in all_epochs.values())

        fig = plt.figure(figsize=(18, 7))
        fig.suptitle("Val F1 Mean — All Cases Comparison",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1.4, 0.35],
                               figure=fig, wspace=0.08)
        ax_main   = fig.add_subplot(gs[0, 0])
        ax_zoom   = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2])
        ax_legend.axis("off")

        handles = []
        for j, (cname, f1_vals) in enumerate(all_f1_mean.items()):
            short = cname.split("_", 1)[1] if "_" in cname else cname
            color = cmap(j % 10)
            ep    = all_epochs[cname]

            line, = ax_main.plot(ep, f1_vals, color=color, linewidth=1.5)
            handles.append(line)

            zoom_ep = [e for e in ep if e >= 25]
            zoom_f1 = [f for e, f in zip(ep, f1_vals) if e >= 25]
            ax_zoom.plot(zoom_ep, zoom_f1, color=color, linewidth=1.8)

        # Main plot
        ax_main.set_xlabel("Epoch", fontsize=11)
        ax_main.set_ylabel("Val F1 Mean", fontsize=11)
        ax_main.set_title("Full Training Curve", fontsize=11)
        ax_main.set_ylim(0, 1.05)
        ax_main.set_xlim(-2, max_ep + 2)
        ax_main.grid(True, alpha=0.3)
        rect = Rectangle((25, y_min), max_ep - 23, y_max - y_min,
                         linewidth=1.2, edgecolor="gray",
                         facecolor="none", linestyle="--", alpha=0.7)
        ax_main.add_patch(rect)
        ax_main.text(26, y_min + 0.002, "zoomed ->",
                     fontsize=7, color="gray", va="bottom")

        # Zoom plot
        ax_zoom.set_xlabel("Epoch", fontsize=11)
        ax_zoom.set_ylabel("Val F1 Mean", fontsize=11)
        ax_zoom.set_title("Converged Region (Epoch ≥ 25)", fontsize=11)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_xlim(25, max_ep + 2)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.yaxis.set_label_position("right")
        ax_zoom.yaxis.tick_right()

        # Legend panel
        short_names = [c.split("_", 1)[1] if "_" in c else c
                       for c in all_f1_mean]
        ax_legend.legend(
            handles=handles, labels=short_names,
            title="Ablation Cases", title_fontsize=9,
            fontsize=8.5, loc="center left",
            frameon=True, framealpha=0.9,
            edgecolor="#cccccc", borderpad=0.8, labelspacing=0.55,
        )

        plt.tight_layout()
        path = os.path.join(curves_dir, "all_cases_f1_overlay.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  All-case F1 overlay saved.")
    elif len(all_f1_mean) == 1:
        print("  Only one non-excluded case — overlay skipped (needs 2+).")

    print(f"\nTraining curves saved to: {curves_dir}")


def plot_pr_roc_curves(all_probs, all_labels, test_runs_dir):
    """
    Computes and saves Precision-Recall and ROC curves for the shoreline class.
    Uses sklearn for threshold sweep; no dependency on model internals.
    """
    from sklearn.metrics import (
        precision_recall_curve, roc_curve,
        average_precision_score, roc_auc_score
    )

    ap   = average_precision_score(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    fpr, tpr, _          = roc_curve(all_labels, all_probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Shoreline Class — PR Curve and ROC Curve",
                 fontsize=13, fontweight="bold")

    # ── PR Curve ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(recall, precision, color="#4C72B0", linewidth=2,
            label=f"PR Curve (AP = {ap:.4f})")
    # Baseline: random classifier at shoreline pixel prevalence
    baseline = all_labels.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1.0,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── ROC Curve ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(fpr, tpr, color="#C44E52", linewidth=2,
            label=f"ROC Curve (AUC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            linewidth=1.0, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(test_runs_dir, "graph_pr_roc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}  (AP={ap:.4f}, AUC={auroc:.4f})")

    # Also save the numeric values to CSV for your thesis table
    curve_csv = os.path.join(test_runs_dir, "pr_roc_scores.csv")
    with open(curve_csv, "w") as f:
        f.write("metric,value\n")
        f.write(f"average_precision,{ap:.6f}\n")
        f.write(f"roc_auc,{auroc:.6f}\n")
    print(f"Saved: {curve_csv}")


def plot_pr_roc_overlay(pr_roc_dict, test_runs_dir):
    """
    Overlays PR and ROC curves for all cases on two charts for direct comparison.
    pr_roc_dict: {case_name: {"precision": arr, "recall": arr,
                               "fpr": arr, "tpr": arr, "ap": float, "auc": float}}
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, \
        average_precision_score, roc_auc_score

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("All Cases — PR and ROC Curve Comparison",
                 fontsize=13, fontweight="bold")

    for i, (case_name, data) in enumerate(pr_roc_dict.items()):
        short = case_name.split("_", 1)[1] if "_" in case_name else case_name
        color = cmap(i)

        axes[0].plot(data["recall"], data["precision"], color=color,
                     linewidth=1.5, label=f"{short} (AP={data['ap']:.3f})")
        axes[1].plot(data["fpr"],    data["tpr"],       color=color,
                     linewidth=1.5, label=f"{short} (AUC={data['auc']:.3f})")

    for ax, xlabel, ylabel, title in zip(
        axes,
        ["Recall", "False Positive Rate"],
        ["Precision", "True Positive Rate"],
        ["Precision-Recall Curve", "ROC Curve"]
    ):
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    axes[0].axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(test_runs_dir, "graph_pr_roc_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_pr_roc_scores_comparison(test_runs_dir, exclude=None):
    """
    Line graph comparing AP and AUC across all cases.
    Reads pr_roc_scores.csv from each case subdirectory — no inference needed.
    Works with any existing test run that already has those CSV files.
    """
    import csv as csv_mod

    if exclude is None:
        exclude = GRAPH_EXCLUDE or []

    summary_path = os.path.join(test_runs_dir, "summary_all_cases.csv")
    if os.path.exists(summary_path):
        case_names = []
        with open(summary_path, newline="") as f:
            for row in csv_mod.DictReader(f):
                case_names.append(row["case"])
    else:
        case_names = sorted([
            d for d in os.listdir(test_runs_dir)
            if os.path.isfile(os.path.join(test_runs_dir, d, "pr_roc_scores.csv"))
        ])

    included = [c for c in case_names if c not in exclude]
    if not included:
        print("  [SKIP pr_roc comparison] No cases to plot after exclusions.")
        return

    ap_values   = []
    auc_values  = []
    valid_cases = []

    for case_name in included:
        csv_path = os.path.join(test_runs_dir, case_name, "pr_roc_scores.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] No pr_roc_scores.csv for {case_name}")
            continue
        scores = {}
        with open(csv_path, newline="") as f:
            for row in csv_mod.DictReader(f):
                scores[row["metric"]] = float(row["value"])
        ap_values.append(scores.get("average_precision", 0.0))
        auc_values.append(scores.get("roc_auc", 0.0))
        valid_cases.append(case_name)

    if not valid_cases:
        print("  [SKIP pr_roc comparison] No pr_roc_scores.csv files found.")
        return

    short_names = [c.split("_", 1)[1] if "_" in c else c for c in valid_cases]
    x = np.arange(len(valid_cases))

    fig, ax = plt.subplots(figsize=(max(10, len(valid_cases) * 1.4), 6))
    fig.suptitle("PR / ROC Summary — All Cases", fontsize=13, fontweight="bold")

    ax.plot(x, ap_values,  "o-", color="#4C72B0", linewidth=2,
            markersize=7, label="Avg Precision (AP)")
    ax.plot(x, auc_values, "s-", color="#C44E52", linewidth=2,
            markersize=7, label="ROC AUC")

    for xi, (ap, auc) in enumerate(zip(ap_values, auc_values)):
        ax.text(xi, ap  + 0.012, f"{ap:.3f}",  ha="center", va="bottom",
                fontsize=8, color="#4C72B0")
        ax.text(xi, auc - 0.022, f"{auc:.3f}", ha="center", va="top",
                fontsize=8, color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(test_runs_dir, "graph_pr_roc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_summary_csv(results_dict, test_runs_dir):
    """Save a single summary CSV with all cases as rows for easy comparison."""
    metrics = ["test_loss", "dice_loss", "ce_loss",
               "accuracy", "precision", "recall",
               "boundary_f1", "mIoU", "SCR"]

    path = os.path.join(test_runs_dir, "summary_all_cases.csv")
    with open(path, "w") as f:
        f.write("case," + ",".join(metrics) + "\n")
        for case_name, res in results_dict.items():
            row = [case_name] + [f"{res[m]:.6f}" for m in metrics]
            f.write(",".join(row) + "\n")
    print(f"\nSummary CSV saved to: {path}")


def regenerate_graphs_from_csv(summary_csv_path, output_dir=None, exclude=None):
    """
    Regenerates the all-case F1 overlay from a previous run's
    summary_all_cases.csv — no model inference needed.

    summary_csv_path : path to an existing summary_all_cases.csv
    output_dir       : folder to write PNGs into. Defaults to the same
                       folder as the CSV so graphs land alongside it.
    exclude          : cases to omit from graphs. Defaults to GRAPH_EXCLUDE.
    """
    import csv as csv_mod

    if exclude is None:
        exclude = GRAPH_EXCLUDE or []

    if not os.path.exists(summary_csv_path):
        print(f"[regenerate] ERROR: file not found: {summary_csv_path}")
        return

    if output_dir is None:
        output_dir = os.path.dirname(summary_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load summary CSV into results_dict ────────────────────────────────
    results_dict = {}
    with open(summary_csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            case_name = row["case"]
            results_dict[case_name] = {k: float(v) for k, v in row.items() if k != "case"}

    if not results_dict:
        print("[regenerate] ERROR: summary CSV is empty.")
        return

    included = [c for c in results_dict if c not in exclude]
    print(f"[regenerate] Loaded {len(results_dict)} cases from: {summary_csv_path}")
    if exclude:
        print(f"[regenerate] Excluding from graphs: {exclude}")
    print(f"[regenerate] Included in graphs   : {included}")
    print(f"[regenerate] Writing graphs to    : {output_dir}")

    # ── All-case F1 overlay from training metrics.csv files ───────────────
    import csv as csv_mod2

    curves_dir = os.path.join(output_dir, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)

    cmap        = plt.get_cmap("tab10")
    all_epochs  = {}
    all_f1_mean = {}

    for i, case_name in enumerate(included):
        csv_path = os.path.join(_EFFECTIVE_TRAIN_RUNS_DIR, case_name, "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP overlay] No metrics.csv for {case_name}")
            continue

        with open(csv_path, newline="") as f:
            reader       = csv_mod2.DictReader(f)
            fieldnames   = reader.fieldnames or []
            has_split_f1 = "val_f1_mean" in fieldnames

            epochs  = []
            f1_mean = []
            for row in reader:
                epochs.append(int(row["epoch"]))
                f1_mean.append(float(row["val_f1_mean"] if has_split_f1 else row["val_f1"]))

        all_epochs[case_name]  = epochs
        all_f1_mean[case_name] = f1_mean

    if len(all_f1_mean) > 1:
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Rectangle

        all_final_f1 = [f[-1] for f in all_f1_mean.values()]
        y_min = min(all_final_f1) - 0.02
        y_max = max(all_final_f1) + 0.015
        max_ep = max(ep[-1] for ep in all_epochs.values())

        fig = plt.figure(figsize=(18, 7))
        fig.suptitle("Val F1 Mean — All Cases Comparison",
                     fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1.4, 0.35],
                               figure=fig, wspace=0.08)
        ax_main   = fig.add_subplot(gs[0, 0])
        ax_zoom   = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 2])
        ax_legend.axis("off")

        handles = []
        for j, (cname, f1_vals) in enumerate(all_f1_mean.items()):
            short = cname.split("_", 1)[1] if "_" in cname else cname
            color = cmap(j % 10)
            ep    = all_epochs[cname]

            line, = ax_main.plot(ep, f1_vals, color=color, linewidth=1.5)
            handles.append(line)

            zoom_ep = [e for e in ep if e >= 25]
            zoom_f1 = [f for e, f in zip(ep, f1_vals) if e >= 25]
            ax_zoom.plot(zoom_ep, zoom_f1, color=color, linewidth=1.8)

        # Main plot
        ax_main.set_xlabel("Epoch", fontsize=11)
        ax_main.set_ylabel("Val F1 Mean", fontsize=11)
        ax_main.set_title("Full Training Curve", fontsize=11)
        ax_main.set_ylim(0, 1.05)
        ax_main.set_xlim(-2, max_ep + 2)
        ax_main.grid(True, alpha=0.3)
        rect = Rectangle((25, y_min), max_ep - 23, y_max - y_min,
                         linewidth=1.2, edgecolor="gray",
                         facecolor="none", linestyle="--", alpha=0.7)
        ax_main.add_patch(rect)
        ax_main.text(26, y_min + 0.002, "zoomed ->",
                     fontsize=7, color="gray", va="bottom")

        # Zoom plot
        ax_zoom.set_xlabel("Epoch", fontsize=11)
        ax_zoom.set_ylabel("Val F1 Mean", fontsize=11)
        ax_zoom.set_title("Converged Region (Epoch ≥ 25)", fontsize=11)
        ax_zoom.set_ylim(y_min, y_max)
        ax_zoom.set_xlim(25, max_ep + 2)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.yaxis.set_label_position("right")
        ax_zoom.yaxis.tick_right()

        # Legend panel
        short_names = [c.split("_", 1)[1] if "_" in c else c
                       for c in all_f1_mean]
        ax_legend.legend(
            handles=handles, labels=short_names,
            title="Ablation Cases", title_fontsize=9,
            fontsize=8.5, loc="center left",
            frameon=True, framealpha=0.9,
            edgecolor="#cccccc", borderpad=0.8, labelspacing=0.55,
        )

        plt.tight_layout()
        path = os.path.join(curves_dir, "all_cases_f1_overlay.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[regenerate] Saved: {path}")
    elif len(all_f1_mean) == 1:
        print("[regenerate] Only one included case has metrics.csv — overlay needs 2+.")
    else:
        print("[regenerate] No metrics.csv files found for included cases — overlay skipped.")
    plot_pr_roc_scores_comparison(output_dir, exclude=exclude)
    print("[regenerate] Done.")


# ── TEST ONE CASE ─────────────────────────────────────────────────────────────
def test_model(case_name, case_cfg, test_imgs, test_masks, test_runs_dir):
    selected_bands = case_cfg["indices"]
    compute_fn     = case_cfg["compute_fn"]
    num_bands      = 1 if compute_fn else len(selected_bands)

    model_path = os.path.join(_EFFECTIVE_TRAIN_RUNS_DIR, case_name, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"  [SKIP] No saved model found at {model_path}")
        return None

    case_dir = os.path.join(test_runs_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    # use make_test_loader — builds only one loader over the real test
    # set, with no dummy train/val data and no misleading batch-shape printout.
    test_loader = make_test_loader(
        test_imgs=test_imgs,
        test_masks=test_masks,
        selected_bands=selected_bands,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        compute_fn=compute_fn,
    )

    model = UNet(num_classes=NUM_CLASSES, num_bands=num_bands)

    # config() moves criterion.ce.weight to device internally
    device, criterion = config()
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # safe formatting - use nan if val_f1 is absent instead of
    # letting :.4f crash on the fallback string 'N/A'.
    saved_f1 = checkpoint.get("val_f1", float("nan"))
    print(f"  Loaded: {model_path}  (Val F1 at save: {saved_f1:.4f})")

    return evaluate(model, test_loader, device, criterion, NUM_CLASSES, case_dir)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Regenerate-only mode ──────────────────────────────────────────────
    # If PREVIOUS_SUMMARY_CSV is set, regenerate graphs from that CSV and
    # exit - no model loading, no inference, no dataset needed.
    if PREVIOUS_SUMMARY_CSV is not None:
        print("Regenerate-only mode: generating graphs from previous summary CSV.\n")
        regenerate_graphs_from_csv(PREVIOUS_SUMMARY_CSV)
        print("\nDone.")
        exit(0)

    # ── Normal testing mode ───────────────────────────────────────────────
    print("Model Testing Started!\n")

    _, _, _, _, test_imgs, test_masks = prepare_train_val_path(root_dir=SHORELINE_ROOT)
    print(f"Test images : {len(test_imgs)}")
    print(f"Test masks  : {len(test_masks)}")
    assert len(test_imgs) > 0, "No test images found!"

    test_runs_dir = prepare_test_runs(TEST_RUNS_DIR)
    all_results   = {}

    for case_name, case_cfg in ABLATION_CASES.items():
        if TEST_ONLY is not None and case_name not in TEST_ONLY:
            print(f"Skipping {case_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  Testing: {case_name}")
        print(f"  Bands:   {case_cfg['indices']}  |  Mode: {case_cfg['compute_fn'] or 'raw'}")
        print(f"{'='*60}")

        results = test_model(case_name, case_cfg, test_imgs, test_masks, test_runs_dir)

        if results is not None:
            all_results[case_name] = results

        torch.cuda.empty_cache()

    tested_cases = list(all_results.keys())

    # Graphs: GRAPH_EXCLUDE cases are tested and saved per-case but hidden
    # from all cross-case comparison charts.
    if len(all_results) > 1:
        plot_pr_roc_scores_comparison(test_runs_dir)
        print(f"\n{'='*60}")
        print("  Generating comparison graphs...")
        if GRAPH_EXCLUDE:
            print(f"  Excluding from graphs: {GRAPH_EXCLUDE}")
        print(f"{'='*60}")
        plot_summary_csv(all_results, test_runs_dir)
    elif len(all_results) == 1:
        plot_summary_csv(all_results, test_runs_dir)

    # Training curves generated for every run; overlay respects GRAPH_EXCLUDE
    if tested_cases:
        print(f"\n{'='*60}")
        print("  Generating training curves...")
        print(f"{'='*60}")
        plot_training_curves(tested_cases, test_runs_dir)

    print("\nTesting complete.")
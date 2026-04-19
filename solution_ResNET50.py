"""
ResNet50-only transfer learning pipeline.

This is a single-model variant of `solution_deep.py`:
- same image preprocessing (grayscale -> ROI crop -> CLAHE),
- same train/val/test split strategy,
- same two-phase training schedule,
- same weighted-F1 evaluation.

Run:
    python solution_ResNET50.py
"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


RANDOM_STATE = 42
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

EPOCHS_HEAD = 6
EPOCHS_FT = 10
LR_HEAD = 1e-3
LR_FT = 1e-4
WEIGHT_DECAY = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_OUTPUT_DIR = "models"
MODEL_OUTPUT_NAME = "resnet50_final.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def create_training_dataframe(train_root: Path) -> pd.DataFrame:
    train_root = train_root.resolve()
    if not train_root.exists() or not train_root.is_dir():
        raise FileNotFoundError(f"TRAIN_SET folder not found: {train_root}")

    class_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    rows = []
    for class_dir in class_dirs:
        label = class_dir.name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() == ".bmp":
                rows.append(
                    {
                        "file_path": str(file_path),
                        "relative_path": str(file_path.relative_to(train_root)),
                        "label": label,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["label", "relative_path"]).reset_index(drop=True)
    return df


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_preprocessed_gray(path: str) -> np.ndarray:
    """Load -> grayscale -> ROI crop -> CLAHE. Returns uint8 HxW."""
    # cv2.imread can fail on Windows paths with non-ASCII characters.
    file_bytes = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[10:532, 92:617]
    gray = _CLAHE.apply(gray)
    return gray


class EyeImageDataset(Dataset):
    """Loads preprocessed grayscale images, replicates to 3 channels, augments."""

    def __init__(self, paths: list[str], labels: list[int], train: bool):
        self.paths = paths
        self.labels = labels

        if train:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
                    transforms.RandomCrop(IMG_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        gray = load_preprocessed_gray(self.paths[idx])
        rgb = np.stack([gray, gray, gray], axis=-1)
        x = self.tf(rgb)
        y = self.labels[idx]
        return x, y


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze(params) -> None:
    for p in params:
        p.requires_grad = True


def build_resnet50(num_classes: int):
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    _freeze_all(model)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    head_params = list(model.fc.parameters())
    ft_params = list(model.layer4.parameters()) + head_params
    return model, head_params, ft_params


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        ps.append(pred)
        ys.append(y.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return f1_score(y_true, y_pred, average="weighted"), y_true, y_pred


def train_phase(
    model,
    loader_tr,
    loader_val,
    device,
    params,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor,
    phase_name: str,
):
    if epochs <= 0 or not params:
        return 0.0, copy.deepcopy(model.state_dict())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, n = 0.0, 0

        for x, y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

        scheduler.step()
        val_f1, _, _ = evaluate(model, loader_val, device)
        print(
            f"    [{phase_name}] epoch {ep:2d}/{epochs}  "
            f"loss={tot_loss / max(n, 1):.4f}  val_f1={val_f1:.4f}  "
            f"({time.time() - t0:.1f}s)"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    return best_f1, best_state


def train_resnet50(
    num_classes: int,
    loader_tr,
    loader_val,
    device,
    class_weights: torch.Tensor,
):
    print("\n=== ResNet50 ===")
    set_seed(RANDOM_STATE)

    model, head_params, ft_params = build_resnet50(num_classes)
    model = model.to(device)

    best_f1_head, best_state = train_phase(
        model,
        loader_tr,
        loader_val,
        device,
        head_params,
        EPOCHS_HEAD,
        LR_HEAD,
        class_weights,
        "head",
    )

    model.load_state_dict(best_state)
    _unfreeze(ft_params)

    best_f1_ft, best_state = train_phase(
        model,
        loader_tr,
        loader_val,
        device,
        ft_params,
        EPOCHS_FT,
        LR_FT,
        class_weights,
        "ft",
    )

    best_f1 = max(best_f1_head, best_f1_ft)
    model.load_state_dict(best_state)
    print(f"  -> best val weighted-F1 = {best_f1:.4f}")
    return model, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_STATE)
    device = pick_device()
    print(f"Device: {device}")

    train_set_path = Path(__file__).resolve().parent / "TRAIN_SET"
    df = create_training_dataframe(train_set_path)
    print(f"Total images: {len(df)}")

    classes = sorted(df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    df["y"] = df["label"].map(class_to_idx)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["y"],
        random_state=RANDOM_STATE,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.20,
        stratify=train_df["y"],
        random_state=RANDOM_STATE,
    )
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    ds_tr = EyeImageDataset(train_df["file_path"].tolist(), train_df["y"].tolist(), train=True)
    ds_val = EyeImageDataset(val_df["file_path"].tolist(), val_df["y"].tolist(), train=False)
    ds_te = EyeImageDataset(test_df["file_path"].tolist(), test_df["y"].tolist(), train=False)

    pin = device.type == "cuda"
    loader_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    loader_te = DataLoader(
        ds_te,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    counts = np.bincount(train_df["y"].to_numpy(), minlength=len(classes))
    cw = counts.sum() / (len(classes) * np.maximum(counts, 1))
    class_weights = torch.tensor(cw, dtype=torch.float32)

    print(f"Class counts (train): {counts.tolist()}")
    print(f"Class weights       : {[round(float(x), 3) for x in cw]}")

    model, best_val_f1 = train_resnet50(
        len(classes), loader_tr, loader_val, device, class_weights
    )

    test_f1, y_true, y_pred = evaluate(model, loader_te, device)
    print("\nClassification report (test set):")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print(f"Weighted F1-score on test set (ResNet50): {test_f1:.4f}")
    print(f"Best validation weighted-F1 (ResNet50): {best_val_f1:.4f}")

    output_dir = Path(__file__).resolve().parent / MODEL_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / MODEL_OUTPUT_NAME

    checkpoint = {
        "arch": "resnet50",
        "weights": ResNet50_Weights.IMAGENET1K_V2.name,
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "img_size": IMG_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "best_val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved trained model checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()

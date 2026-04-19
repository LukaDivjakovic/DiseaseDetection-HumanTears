"""
Inference script for the trained ResNet50 model.

Usage:
    python solution.py <path_to_test_set>

The test set path can be either:
  - a directory containing .bmp images (searched recursively), or
  - a path to a single .bmp image.

For each .bmp image found, prints a line:
    <picture path>: <predicted class>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


MODEL_PATH = Path(__file__).resolve().parent / "models" / "resnet50_final.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_preprocessed_gray(path: str) -> np.ndarray:
    """Load -> grayscale -> ROI crop -> CLAHE. Matches training preprocessing."""
    file_bytes = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[10:532, 92:617]
    gray = _CLAHE.apply(gray)
    return gray


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def collect_bmp_paths(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() == ".bmp":
            return [root]
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".bmp")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python solution.py <path_to_test_set>", file=sys.stderr)
        sys.exit(1)

    test_path = Path(sys.argv[1]).resolve()
    if not test_path.exists():
        print(f"Path not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    device = pick_device()

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    img_size = checkpoint.get("img_size", 224)

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    paths = collect_bmp_paths(test_path)
    if not paths:
        print(f"No .bmp images found at: {test_path}", file=sys.stderr)
        sys.exit(1)

    with torch.no_grad():
        for p in paths:
            gray = load_preprocessed_gray(str(p))
            rgb = np.stack([gray, gray, gray], axis=-1)
            x = tf(rgb).unsqueeze(0).to(device)
            logits = model(x)
            pred_idx = int(logits.argmax(dim=1).item())
            print(f"{p}: {classes[pred_idx]}")


if __name__ == "__main__":
    main()

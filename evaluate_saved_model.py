from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from solution_efficientnet import (
    RANDOM_STATE,
    create_training_dataframe,
    load_images,
)
from solution_ResNET50 import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_preprocessed_gray,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved .keras or .pth model on the same held-out test split used in training."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing TRAIN_SET and models directories.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a .keras or .pth model file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Load models/fold_<N>_model.keras from project-dir/models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used for prediction.",
    )
    return parser.parse_args()


def resolve_model_path(project_dir: Path, model_path: Path | None, fold: int | None) -> Path:
    if model_path is not None:
        resolved = model_path if model_path.is_absolute() else (project_dir / model_path)
        return resolved.resolve()

    if fold is not None:
        if fold < 1:
            raise ValueError("--fold must be >= 1")
        return (project_dir / "models" / f"fold_{fold}_model.keras").resolve()

    return (project_dir / "models" / "final_model.keras").resolve()


def pick_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchEvaluationDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], img_size: int):
        self.paths = paths
        self.labels = labels
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
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


def build_resnet50_from_checkpoint(checkpoint: dict) -> nn.Module:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes = checkpoint.get("num_classes")

    if num_classes is None and isinstance(state_dict, dict) and "fc.weight" in state_dict:
        num_classes = int(state_dict["fc.weight"].shape[0])

    if num_classes is None:
        raise ValueError("Could not determine number of classes from the .pth checkpoint.")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_pytorch_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
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


def evaluate_saved_model(project_dir: Path, model_path: Path, batch_size: int) -> None:
    train_set_path = (project_dir / "TRAIN_SET").resolve()
    train_df = create_training_dataframe(train_set_path)
    if train_df.empty:
        raise ValueError("No training images were found in TRAIN_SET.")

    _, test_part = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["label"],
        random_state=RANDOM_STATE,
    )

    print(f"Loading model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    suffix = model_path.suffix.lower()
    if suffix == ".keras":
        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["label"])
        y_test = label_encoder.transform(test_part["label"])
        class_names = list(label_encoder.classes_)

        model = tf.keras.models.load_model(str(model_path))

        print("Loading test images...")
        X_test = load_images(test_part, augment=False)
        print(f"Test samples: {len(X_test)}")

        y_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        print("\nClassification report (test set):")
        print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
        print(f"Weighted F1-score on test set: {weighted_f1:.4f}")
        return

    if suffix == ".pth":
        checkpoint = torch.load(str(model_path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a checkpoint dictionary in the .pth file.")

        class_names = list(checkpoint.get("classes") or sorted(train_df["label"].unique().tolist()))
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)
        y_test = label_encoder.transform(test_part["label"])

        eval_img_size = int(checkpoint.get("img_size", IMG_SIZE))
        device = pick_torch_device()
        print(f"Using PyTorch device: {device}")

        model = build_resnet50_from_checkpoint(checkpoint).to(device)

        test_dataset = TorchEvaluationDataset(
            test_part["file_path"].tolist(),
            y_test.tolist(),
            img_size=eval_img_size,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        print("Loading test images...")
        print(f"Test samples: {len(test_dataset)}")
        weighted_f1, y_true, y_pred = evaluate_pytorch_model(model, test_loader, device)

        print("\nClassification report (test set):")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        print(f"Weighted F1-score on test set: {weighted_f1:.4f}")
        return

    raise ValueError(f"Unsupported model format: {model_path.suffix}. Use a .keras or .pth file.")


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    model_path = resolve_model_path(project_dir, args.model_path, args.fold)
    evaluate_saved_model(project_dir=project_dir, model_path=model_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()


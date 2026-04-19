from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from solution_efficientnet import (
    RANDOM_STATE,
    create_training_dataframe,
    load_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved .keras model on the same held-out test split used in training."
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
        help="Path to a .keras model file (e.g., models/final_model.keras).",
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

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])
    y_test = label_encoder.transform(test_part["label"])
    class_names = list(label_encoder.classes_)

    print(f"Loading model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
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


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    model_path = resolve_model_path(project_dir, args.model_path, args.fold)
    evaluate_saved_model(project_dir=project_dir, model_path=model_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()


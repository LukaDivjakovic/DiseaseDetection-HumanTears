from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report


HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    block_norm="L2-Hys",
)


def create_training_dataframe(train_root: Path) -> pd.DataFrame:
    train_root = train_root.resolve()
    if not train_root.exists() or not train_root.is_dir():
        raise FileNotFoundError(f"TRAIN_SET folder not found: {train_root}")

    class_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    if len(class_dirs) != 5:
        raise ValueError(
            f"Expected 5 class subdirectories in {train_root}, found {len(class_dirs)}"
        )

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


def load_and_preprocess(path: str) -> np.ndarray:
    """Load image, convert to grayscale, crop."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    img = img[10:532, 92:617]
    return img


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (3, 3),
        block_norm: str = "L2-Hys",
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = [
            hog(
                img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                feature_vector=True,
            )
            for img in X
        ]
        return np.vstack(feats)


def load_images(df: pd.DataFrame, augment: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load (and optionally augment) images. Returns (images, labels).

    images is a 1D np.ndarray of dtype=object so sklearn indexing works during CV.
    """
    images = []
    labels = []
    for fp, lbl in zip(df["file_path"].tolist(), df["label"].tolist()):
        img = load_and_preprocess(fp)
        images.append(img)
        labels.append(lbl)
        if augment:
            images.append(rotate_image(img, 20))
            labels.append(lbl)
    X = np.empty(len(images), dtype=object)
    for i, im in enumerate(images):
        X[i] = im
    return X, np.array(labels)


if __name__ == "__main__":
    train_set_path = Path(__file__).resolve().parent / "TRAIN_SET"
    train_df = create_training_dataframe(train_set_path)

    print("DataFrame created successfully")
    print(f"Total images: {len(train_df)}")
    print(f"Labels: {sorted(train_df['label'].unique()) if not train_df.empty else []}")
    print(train_df.head())

    output_file = Path(__file__).resolve().parent / "train_set_dataframe.csv"
    train_df.to_csv(output_file, index=False)
    print(f"Saved CSV: {output_file}")

    # Train/test split (stratified)
    train_part, test_part = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["label"],
        random_state=42,
    )
    print(f"Train size: {len(train_part)}, Test size: {len(test_part)}")

    # Load images (augment training set with 20° rotation)
    print("Loading training images (with augmentation)...")
    X_train, y_train = load_images(train_part, augment=True)
    print("Loading test images...")
    X_test, y_test = load_images(test_part, augment=False)
    print(f"Train samples (after augmentation): {len(X_train)}, Test samples: {len(X_test)}")

    # Pipeline: HOG -> Scaler -> SVM
    pipe = Pipeline(
        [
            ("hog", HOGTransformer()),
            ("scaler", StandardScaler()),
            ("svm", SVC()),
        ]
    )

    # Search space: HOG + SVM params.
    # NOTE: very small pixels_per_cell combined with large cells_per_block
    # blows up the HOG feature vector size and causes RAM to explode when
    # many CV folds run in parallel. We keep only combinations that produce
    # a reasonable feature size.
    param_dist = {
        "hog__orientations": [6, 8, 9, 10, 12],
        "hog__pixels_per_cell": [(8, 8), (10, 10), (12, 12), (16, 16)],
        "hog__cells_per_block": [(2, 2), (3, 3)],
        "hog__block_norm": ["L1", "L2", "L2-Hys"],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"],
        "svm__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Limit parallelism so we don't hold many copies of the (augmented)
    # training set + HOG features in memory at once. `pre_dispatch` caps
    # how many jobs joblib queues up ahead of time.
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=2,
        pre_dispatch="2*n_jobs",
        verbose=2,
        error_score=np.nan,
        refit=True,
    )

    print("\nRunning RandomizedSearchCV over HOG + SVM params...")
    search.fit(X_train, y_train)

    print(f"\nBest CV weighted F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Evaluate best model on held-out test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Weighted F1-score on test set: {weighted_f1:.4f}")

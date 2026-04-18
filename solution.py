from pathlib import Path

import pandas as pd


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

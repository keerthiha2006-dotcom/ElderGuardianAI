"""Train and evaluate a fall-state classifier from a UR Fall CSV export.

CSV format: video_id, frame_number, label, feature_1 ... feature_8
Labels are -1, 0, and 1. Whole videos are split together to prevent leakage.
"""

from pathlib import Path
import argparse
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMNS = ["video_id", "frame_number", "label"] + [f"feature_{i}" for i in range(1, 9)]


def train(dataset_path, model_path, report_path):
    data = pd.read_csv(dataset_path, header=None, names=COLUMNS)
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = [column for column in COLUMNS if column.startswith("feature_")]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_indices, test_indices = next(splitter.split(data[features], data["label"], data["video_id"]))
    train_data, test_data = data.iloc[train_indices], data.iloc[test_indices]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(train_data[features], train_data["label"])
    predictions = model.predict(test_data[features])

    report = classification_report(
        test_data["label"], predictions, labels=[-1, 0, 1], output_dict=True,
        zero_division=0,
    )
    result = {
        "dataset": str(dataset_path),
        "rows": int(len(data)),
        "features": features,
        "labels": [-1, 0, 1],
        "train_videos": sorted(train_data["video_id"].unique().tolist()),
        "test_videos": sorted(test_data["video_id"].unique().tolist()),
        "classification_report": report,
        "confusion_matrix_labels": [-1, 0, 1],
        "confusion_matrix": confusion_matrix(
            test_data["label"], predictions, labels=[-1, 0, 1]
        ).tolist(),
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "labels": [-1, 0, 1]}, model_path)
    Path(report_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": str(model_path),
        "report": str(report_path),
        "rows": len(data),
        "accuracy": report["accuracy"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="urfall-cam0-falls.csv")
    parser.add_argument("--model", default="models/fall_model.joblib")
    parser.add_argument("--report", default="models/fall_model_report.json")
    args = parser.parse_args()
    train(args.dataset, args.model, args.report)

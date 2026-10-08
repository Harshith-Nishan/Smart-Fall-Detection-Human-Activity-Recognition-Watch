import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


BASE_DIR = Path(__file__).parent
DATA_FOLDER = Path(
    os.getenv(
        "HAR_DATA_FOLDER",
        BASE_DIR / "dataset" / "raw_data"
    )
)

WINDOW_SIZE = 40
STEP_SIZE = 20
MAX_DEPTH = 12


def extract_esp32_features(window):
    features = []

    for i in range(6):
        x = window[:, i]

        features.extend([
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.sqrt(np.mean(x ** 2)),
            np.mean(np.abs(np.diff(x)))
        ])

    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
    acc = np.sqrt(ax * ax + ay * ay + az * az)

    features.extend([
        np.mean(acc),
        np.std(acc),
        np.min(acc),
        np.max(acc),
        np.sqrt(np.mean(acc ** 2)),
        np.mean(np.abs(np.diff(acc)))
    ])

    gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]
    gyro = np.sqrt(gx * gx + gy * gy + gz * gz)

    features.extend([
        np.mean(gyro),
        np.std(gyro),
        np.min(gyro),
        np.max(gyro),
        np.sqrt(np.mean(gyro ** 2)),
        np.mean(np.abs(np.diff(gyro)))
    ])

    def correlation(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    features.extend([
        correlation(ax, ay),
        correlation(ax, az),
        correlation(ay, az),
        correlation(gx, gy),
        correlation(gx, gz),
        correlation(gy, gz)
    ])

    return np.asarray(features, dtype=np.float32)


files = sorted(
    glob.glob(str(DATA_FOLDER / "user*.csv"))
)

print("=" * 60)
print("ESP32 MODEL TRAINING")
print("=" * 60)
print(f"Users found: {len(files)}")

X = []
y = []

for file in files:
    print("Processing:", Path(file).name)

    df = pd.read_csv(file)

    sensors = df[
        ["ax", "ay", "az", "gx", "gy", "gz"]
    ].values

    labels = df["label"].values

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):
        end = start + WINDOW_SIZE

        window = sensors[start:end]
        window_labels = labels[start:end]

        unique_labels = np.unique(window_labels)

        if len(unique_labels) != 1:
            continue

        features = extract_esp32_features(window)

        X.append(features)
        y.append(unique_labels[0])


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

print("\nDataset created")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Features per window:", X.shape[1])


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nClasses:")
for index, label in enumerate(encoder.classes_):
    print(index, "=", label)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


model = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", f"{accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

print("Tree depth:", model.get_depth())
print("Tree nodes:", model.tree_.node_count)


tree_text = export_text(
    model,
    feature_names=[
        f"f{i}"
        for i in range(X.shape[1])
    ]
)

np.save(
    BASE_DIR / "esp32_tree_children_left.npy",
    model.tree_.children_left
)
np.save(
    BASE_DIR / "esp32_tree_children_right.npy",
    model.tree_.children_right
)
np.save(
    BASE_DIR / "esp32_tree_features.npy",
    model.tree_.feature
)
np.save(
    BASE_DIR / "esp32_tree_threshold.npy",
    model.tree_.threshold
)
np.save(
    BASE_DIR / "esp32_tree_values.npy",
    model.tree_.value
)
np.save(
    BASE_DIR / "esp32_classes.npy",
    encoder.classes_
)

(BASE_DIR / "esp32_tree.txt").write_text(
    tree_text,
    encoding="utf-8"
)

info = [
    f"Feature count: {X.shape[1]}",
    f"Tree depth: {model.get_depth()}",
    f"Tree nodes: {model.tree_.node_count}",
    "Classes:"
]

info.extend(
    f"{i}: {label}"
    for i, label in enumerate(encoder.classes_)
)

(BASE_DIR / "esp32_model_info.txt").write_text(
    "\n".join(info) + "\n",
    encoding="utf-8"
)

print("\nGenerated ESP32 model files.")

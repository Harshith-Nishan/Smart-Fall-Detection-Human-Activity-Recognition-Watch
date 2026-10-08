import os
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import serial

from feature_extraction import extract_features


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "advanced_lightgbm_har_model.pkl"
ENCODER_PATH = BASE_DIR / "advanced_activity_label_encoder.pkl"

PORT = os.getenv("ESP32_PORT", "COM3")
BAUDRATE = 115200

WINDOW_SIZE = 40
WINDOWS_PER_ACTIVITY = 20

ACTIVITIES = {
    "1": ("Wa", "Walking"),
    "2": ("J", "Jogging"),
    "3": ("T", "Typing"),
    "4": ("Wr", "Writing"),
    "5": ("C", "Cycling"),
    "6": ("U", "Upstairs"),
    "7": ("D", "Downstairs"),
}


model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

ser = serial.Serial(
    PORT,
    BAUDRATE,
    timeout=2
)

time.sleep(2)
ser.reset_input_buffer()

print("Connected to ESP32:", PORT)


def read_sensor():
    while True:
        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 6:
            continue

        try:
            return [float(value) for value in parts]
        except ValueError:
            continue


def collect_window():
    buffer = []

    while len(buffer) < WINDOW_SIZE:
        buffer.append(read_sensor())
        print(
            f"\rCollecting "
            f"{len(buffer)}/{WINDOW_SIZE}",
            end=""
        )

    print()

    return np.asarray(
        buffer,
        dtype=np.float32
    )


def predict_window(window):
    features = np.asarray(
        extract_features(window),
        dtype=np.float32
    ).reshape(1, -1)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    activity = encoder.inverse_transform([prediction])[0]
    confidence = np.max(probabilities)

    return activity, confidence


all_actual = []
all_predicted = []

try:
    print("\nReal-world ESP32 Human Activity Recognition Test")
    print(f"Windows per activity: {WINDOWS_PER_ACTIVITY}")
    print("Each window: 40 samples = 2 seconds")

    for _, (label, name) in ACTIVITIES.items():
        print("\n" + "=" * 60)
        print("Activity:", name)
        print("Expected label:", label)
        print("=" * 60)

        input(f"Perform {name} and press ENTER to start...")

        predictions = []
        confidences = []

        for i in range(WINDOWS_PER_ACTIVITY):
            print(
                f"\nWindow {i + 1}/{WINDOWS_PER_ACTIVITY}"
            )

            window = collect_window()
            predicted, confidence = predict_window(window)

            predictions.append(predicted)
            confidences.append(confidence)

            all_actual.append(label)
            all_predicted.append(predicted)

            print(
                "Prediction:",
                predicted,
                "✓" if predicted == label else "✗"
            )
            print(
                f"Confidence: {confidence * 100:.2f}%"
            )

        correct_count = sum(
            prediction == label
            for prediction in predictions
        )

        accuracy = (
            correct_count / WINDOWS_PER_ACTIVITY
        ) * 100

        average_confidence = np.mean(confidences) * 100
        most_common = Counter(predictions).most_common(1)[0][0]

        print("\nResult:", name)
        print(
            f"Correct predictions: "
            f"{correct_count}/{WINDOWS_PER_ACTIVITY}"
        )
        print(f"Accuracy: {accuracy:.2f}%")
        print(
            f"Average confidence: "
            f"{average_confidence:.2f}%"
        )
        print(f"Most predicted: {most_common}")

    all_actual = np.asarray(all_actual)
    all_predicted = np.asarray(all_predicted)

    overall_accuracy = (
        np.mean(all_actual == all_predicted) * 100
    )

    print("\n" + "=" * 60)
    print("Final Real-World Results")
    print("=" * 60)
    print(f"Total windows tested: {len(all_actual)}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")

    labels = list(encoder.classes_)
    matrix = np.zeros(
        (len(labels), len(labels)),
        dtype=int
    )

    for actual, predicted in zip(
        all_actual,
        all_predicted
    ):
        actual_index = labels.index(actual)
        predicted_index = labels.index(predicted)
        matrix[actual_index, predicted_index] += 1

    print("\nConfusion Matrix:")
    print(matrix)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ser.close()
    print("ESP32 connection closed.")

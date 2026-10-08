import os
import time
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
STEP_SIZE = 20


print("Loading model...")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("Model loaded successfully.")
print("Connecting to ESP32...")

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


buffer = []

try:
    print("\nLive Human Activity Recognition")
    print("Sampling at approximately 20 Hz")
    print("Window: 40 samples (2 seconds)")
    print("Overlap: 50%\n")

    while True:
        while len(buffer) < WINDOW_SIZE:
            buffer.append(read_sensor())
            print(
                f"\rCollecting samples: "
                f"{len(buffer)}/{WINDOW_SIZE}",
                end=""
            )

        print()

        window = np.asarray(
            buffer,
            dtype=np.float32
        )

        if window.shape != (WINDOW_SIZE, 6):
            buffer = []
            continue

        features = np.asarray(
            extract_features(window),
            dtype=np.float32
        ).reshape(1, -1)

        if features.shape[1] != 120:
            raise ValueError(
                f"Expected 120 features, got {features.shape[1]}"
            )

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        activity = encoder.inverse_transform([prediction])[0]

        print("\n" + "=" * 50)
        print("Predicted activity:", activity)
        print("=" * 50)

        for label, probability in zip(
            encoder.classes_,
            probabilities
        ):
            print(f"{label:5s}: {probability * 100:6.2f}%")

        buffer = buffer[STEP_SIZE:]

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ser.close()
    print("ESP32 connection closed.")

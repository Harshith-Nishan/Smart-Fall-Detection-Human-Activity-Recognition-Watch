# ESP32 Human Activity Recognition using IMU

A real-time Human Activity Recognition system using an ESP32 and MPU6050 IMU sensor.

The system collects 6-axis motion data (3-axis accelerometer and 3-axis gyroscope), extracts statistical, magnitude, correlation, and frequency-domain features, and uses a lightweight Decision Tree model for activity classification.

The model recognizes seven activities:

- Cycling
- Downstairs
- Jogging
- Typing
- Upstairs
- Walking
- Writing

The trained model is converted into a lightweight C++ representation for ESP32 deployment, allowing activity classification with minimal computational requirements.

## System Flow

MPU6050
↓
ESP32
↓
6-Axis Motion Data
↓
Feature Extraction
↓
Decision Tree Model
↓
Activity Classification

## Technologies

- ESP32
- MPU6050
- Python
- NumPy
- Scikit-learn
- Decision Tree
- C/C++
- Serial Communication

## Model

- Features: 54
- Maximum tree depth: 12
- Tree nodes: 637
- Classes: 7
- Sampling rate: 20 Hz
- Window size: 40 samples
- Window duration: 2 seconds

## Deployment

The trained Decision Tree is converted into an ESP32-compatible C++ model, allowing the classifier to run efficiently on the embedded device.

## Future Development

The system can be extended toward wearable fall detection by adding dedicated fall-event detection, emergency alerts, and communication with a mobile or cloud platform.

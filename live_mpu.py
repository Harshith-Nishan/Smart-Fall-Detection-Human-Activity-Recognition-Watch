import os
import time
import serial


PORT = os.getenv("ESP32_PORT", "COM3")
BAUDRATE = 115200

print("Opening", PORT)

ser = serial.Serial(
    PORT,
    BAUDRATE,
    timeout=2
)

time.sleep(2)

print("Connected to ESP32:", PORT)
print("Reading raw serial lines...")

try:
    while True:
        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        parts = line.split(",")

        if len(parts) != 6:
            print("RECEIVED:", repr(line))
            continue

        try:
            values = [float(value) for value in parts]
            print("VALID SENSOR DATA:", values)
        except ValueError:
            print("Invalid sensor data:", repr(line))

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ser.close()
    print("ESP32 connection closed.")

import numpy as np


def extract_features(window):
    features = []

    for i in range(6):
        signal = window[:, i]

        features.extend([
            np.mean(signal),
            np.std(signal),
            np.min(signal),
            np.max(signal),
            np.ptp(signal),
            np.median(signal),
            np.sqrt(np.mean(signal ** 2)),
            np.sum(signal ** 2),
            np.mean(np.abs(signal)),
            np.mean(np.abs(np.diff(signal))),
            np.ptp(signal)
        ])

    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
    acc = np.sqrt(ax**2 + ay**2 + az**2)

    features.extend([
        np.mean(acc),
        np.std(acc),
        np.min(acc),
        np.max(acc),
        np.sqrt(np.mean(acc**2)),
        np.sum(acc**2),
        np.mean(np.abs(np.diff(acc)))
    ])

    gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]
    gyro = np.sqrt(gx**2 + gy**2 + gz**2)

    features.extend([
        np.mean(gyro),
        np.std(gyro),
        np.min(gyro),
        np.max(gyro),
        np.sqrt(np.mean(gyro**2)),
        np.sum(gyro**2),
        np.mean(np.abs(np.diff(gyro)))
    ])

    features.extend([
        np.corrcoef(ax, ay)[0, 1],
        np.corrcoef(ax, az)[0, 1],
        np.corrcoef(ay, az)[0, 1],
        np.corrcoef(gx, gy)[0, 1],
        np.corrcoef(gx, gz)[0, 1],
        np.corrcoef(gy, gz)[0, 1]
    ])

    fs = 20.0

    for i in range(6):
        signal = window[:, i] - np.mean(window[:, i])
        fft_values = np.abs(np.fft.rfft(signal))
        frequencies = np.fft.rfftfreq(len(signal), d=1 / fs)

        fft_values[0] = 0

        dominant_index = np.argmax(fft_values)
        dominant_frequency = frequencies[dominant_index]
        spectral_energy = np.sum(fft_values ** 2)
        total_magnitude = np.sum(fft_values)

        if total_magnitude > 0:
            spectral_centroid = (
                np.sum(frequencies * fft_values) / total_magnitude
            )
        else:
            spectral_centroid = 0

        low_energy = np.sum(
            fft_values[frequencies <= 2] ** 2
        )
        high_energy = np.sum(
            fft_values[frequencies > 2] ** 2
        )

        features.extend([
            dominant_frequency,
            spectral_energy,
            spectral_centroid,
            low_energy,
            high_energy
        ])

    acc_centered = acc - np.mean(acc)
    acc_fft = np.abs(np.fft.rfft(acc_centered))
    acc_freq = np.fft.rfftfreq(
        len(acc_centered),
        d=1 / fs
    )

    acc_fft[0] = 0
    acc_index = np.argmax(acc_fft)

    features.extend([
        acc_freq[acc_index],
        np.sum(acc_fft ** 2)
    ])

    gyro_centered = gyro - np.mean(gyro)
    gyro_fft = np.abs(np.fft.rfft(gyro_centered))
    gyro_freq = np.fft.rfftfreq(
        len(gyro_centered),
        d=1 / fs
    )

    gyro_fft[0] = 0
    gyro_index = np.argmax(gyro_fft)

    features.extend([
        gyro_freq[gyro_index],
        np.sum(gyro_fft ** 2)
    ])

    return features

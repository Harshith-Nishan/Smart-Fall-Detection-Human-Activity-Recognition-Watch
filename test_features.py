import numpy as np

from feature_extraction import extract_features


window = np.random.randn(40, 6)

features = extract_features(window)

print("Window shape:", window.shape)
print("Number of features:", len(features))

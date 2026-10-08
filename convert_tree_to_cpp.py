import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).parent

children_left = np.load(BASE_DIR / "esp32_tree_children_left.npy")
children_right = np.load(BASE_DIR / "esp32_tree_children_right.npy")
features = np.load(BASE_DIR / "esp32_tree_features.npy")
thresholds = np.load(BASE_DIR / "esp32_tree_threshold.npy")
values = np.load(BASE_DIR / "esp32_tree_values.npy")
classes = np.load(BASE_DIR / "esp32_classes.npy")

node_count = len(children_left)

print("Tree nodes:", node_count)
print("Classes:", classes)
print("Generating C++ model...")

output_path = BASE_DIR / "esp32_model.h"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("#ifndef ESP32_MODEL_H\n")
    f.write("#define ESP32_MODEL_H\n\n")
    f.write("// Automatically generated Decision Tree model\n")
    f.write("// For ESP32 Human Activity Recognition\n\n")

    f.write(f"const int TREE_NODES = {node_count};\n")
    f.write(f"const int TREE_CLASSES = {len(classes)};\n\n")

    f.write("const int16_t TREE_LEFT[] = {\n")
    for i, value in enumerate(children_left):
        f.write(f"    {int(value)},")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n\n")

    f.write("const int16_t TREE_RIGHT[] = {\n")
    for i, value in enumerate(children_right):
        f.write(f"    {int(value)},")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n\n")

    f.write("const int16_t TREE_FEATURE[] = {\n")
    for i, value in enumerate(features):
        f.write(f"    {int(value)},")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n\n")

    f.write("const float TREE_THRESHOLD[] = {\n")
    for i, value in enumerate(thresholds):
        f.write(f"    {float(value):.10f}f,")
        if i % 6 == 5:
            f.write("\n")
    f.write("\n};\n\n")

    f.write("const int TREE_PREDICTION[] = {\n")
    for i in range(node_count):
        prediction = int(np.argmax(values[i][0]))
        f.write(f"    {prediction},")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n\n")

    f.write("const char* ACTIVITY_NAMES[] = {\n")
    for label in classes:
        f.write(f'    "{label}",\n')
    f.write("};\n\n")

    f.write(
        """int predictActivity(float features[])
{
    int node = 0;

    while (TREE_LEFT[node] != -1)
    {
        int featureIndex = TREE_FEATURE[node];
        float value = features[featureIndex];

        if (value <= TREE_THRESHOLD[node])
        {
            node = TREE_LEFT[node];
        }
        else
        {
            node = TREE_RIGHT[node];
        }
    }

    return TREE_PREDICTION[node];
}

"""
    )

    f.write("#endif\n")

print(f"Created: {output_path.name}")

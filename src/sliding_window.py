# sliding_window.py
import numpy as np
import torch
from tensorflow.keras.models import load_model
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os

class_names = [
    "20",
    "30",
    "50",
    "60",
    "70",
    "80",
    "No 80",
    "100",
    "120",
    "Passing",
    "Truck passing",
    "Intersection ahead",
    "Yellow diamond",
    "Yield",
    "Stop",
    "Circle",
    "Truck",
    "No entry",
    "Exclamation",
    "Left turn",
    "Right turn",
    "Squiggle",
    "Speed bumps",
    "Drift",
    "Narrows",
    "Construction",
    "Stoplight",
    "Crosswalk",
    "People running",
    "Bicycle",
    "Snowflake",
    "Deer",
    "No limit",
    "Blue right turn",
    "Blue left turn",
    "Blue forward",
    "Blue forward or right",
    "Blue forward or left",
    "Blue bottom right",
    "Blue bottom left",
    "Blue roundabout",
    "No passing",
    "No truck passing"
]

# Load the model
model = load_model('src/models/best_model_so_far.h5')

window_size = 32
stride = 11

for img_path in Path("test_images").iterdir():
    # Put the color axis last for the stride process
    full_img = cv2.imread(str(img_path)) / 255.0
    windows = torch.tensor(full_img).unfold(0, window_size, stride).unfold(1, window_size, stride).numpy()
    window_axes = windows.shape[:2]

    # Run classification
    batch = windows.reshape(-1, 3, window_size, window_size).transpose(0, 2, 3, 1)
    save_dir = Path.home() / "windows"
    save_dir.mkdir(exist_ok=True)
    for i, img in enumerate(batch):
        # Construct a filename for each image
        filename = f"image_{i}.png"
        file_path = os.path.join(save_dir, filename)
        cv2.imwrite(file_path, (255 * img).astype(np.uint8))

    output = model(batch).numpy()
    output_image = output.argmax(-1).reshape(*window_axes)
    confidences_image = output.max(-1).reshape(*window_axes)

    object_coordinates = np.nonzero(output_image != 43)
    object_pixel_coordinates = list(zip(object_coordinates[0], object_coordinates[1]))
    for i, j in object_pixel_coordinates:
        print("Found", class_names[output_image[i, j]], "at", i, j, "with confidence", confidences_image[i, j])

        half_width = 25
        x = (j * stride) + (window_size // 2)
        y = (i * stride) + (window_size // 2)
        if confidences_image[i, j] > 0.98:
            cv2.rectangle(full_img, (x - half_width, y - half_width), (x + half_width, y + half_width), (0, 0, 255), thickness=5)

    plt.imshow(np.flip(full_img, axis=2))
    plt.show()

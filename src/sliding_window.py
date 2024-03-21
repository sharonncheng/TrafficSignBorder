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

scaled_image_widths = [1500, 1200, 900]

# Load the model
model = load_model('src/models/best_model_so_far.h5')

window_size = 32
stride = 16

for img_idx, img_path in enumerate(sorted(Path("test_images").iterdir())):
    # Put the color axis last for the stride process
    original_img = cv2.imread(str(img_path))
    full_img = original_img / 255.0

    detections = []

    for new_width in scaled_image_widths:
        scale_ratio = new_width / max(full_img.shape)
        original_dims = (full_img.shape[1], full_img.shape[0])
        scaled_dims = (int(round(original_dims[0] * scale_ratio)), int(round(original_dims[1] * scale_ratio)))
        scaled_img = cv2.resize(full_img, scaled_dims)
        print("scaling", full_img.shape, scaled_img.shape, scaled_dims)

        windows = torch.tensor(scaled_img).unfold(0, window_size, stride).unfold(1, window_size, stride).numpy()
        window_axes = windows.shape[:2]

        # Run classification
        batch = windows.reshape(-1, 3, window_size, window_size).transpose(0, 2, 3, 1)
        save_dir = Path.home() / "windows"
        save_dir.mkdir(exist_ok=True)
        for i, img in enumerate(batch):
            filename = f"image_{i}.png"
            file_path = os.path.join(save_dir, filename)
            cv2.imwrite(file_path, (255 * img).astype(np.uint8))

        output = model(batch).numpy()
        output_image = output.argmax(-1).reshape(*window_axes)
        output_dims = (output_image.shape[1], output_image.shape[0])
        confidences_image = output.max(-1).reshape(*window_axes)

        raw_coords_x, raw_coords_y = np.nonzero(output_image != 43)
        confidences = [confidences_image[x, y] for x, y in zip(raw_coords_x, raw_coords_y)]
        classes = [output_image[x, y] for x, y in zip(raw_coords_x, raw_coords_y)]
        # Offset by a half pixel to get the center
        raw_coords_x = raw_coords_x.astype(float) + 0.5
        raw_coords_y = raw_coords_y.astype(float) + 0.5
        # Scale to the original image size
        raw_coords_x *= original_dims[1] / output_dims[1]
        raw_coords_y *= original_dims[0] / output_dims[0]
        for x, y, confidence, obj_class in zip(raw_coords_x, raw_coords_y, confidences, classes, strict=True):
            detections.append((int(round(x)), int(round(y)), confidence, obj_class))
        print(detections)
        print()
        print()
        print()
        print()

    for x, y, confidence, obj_class in detections:

        half_width = 36
        if confidence > 0.8:
            print("Found", class_names[obj_class], "at", x, y, "with confidence", confidence)
            cv2.rectangle(original_img, (y - half_width, x - half_width), (y + half_width, x + half_width), (0, 0, 255), thickness=5)
            cv2.putText(original_img, class_names[obj_class], (y + half_width, x + half_width), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2, 2)

    cv2.imwrite(f"{img_idx}.png", original_img)

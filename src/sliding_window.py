# sliding_window.py
import numpy as np
import torch
from tensorflow.keras.models import load_model
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Load the model
model = load_model('src/models/best_model.h5')

window_size = 32
stride = 11

for img_path in Path("test_images").iterdir():
    # Put the color axis last for the stride process
    img = cv2.imread(str(img_path))
    print(img.shape)
    windows = torch.tensor(img).unfold(0, window_size, stride).unfold(1, window_size, stride).numpy()
    print(windows.shape)
    window_axes = windows.shape[:2]
    print(window_axes)

    # Run classification
    batch = windows.reshape(-1, 3, window_size, window_size).transpose(0, 2, 3, 1)
    output = model(batch).numpy().argmax(-1)
    output_image = output.reshape(*window_axes)
    plt.imshow(error_image)
    plt.show()
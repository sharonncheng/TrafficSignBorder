import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_model
from data_prep import X_train, y_train, X_val, y_val, X_test, y_test
import cv2

# Model parameters
input_shape = X_train.shape[1:]  # e.g., (32, 32, 3) for 32x32 RGB images
num_classes = y_train.shape[1]  # Based on your dataset

model = create_model(input_shape, num_classes)

def load_and_process_images(directory, crop_shape, num_images=8, crops_per_image=16):
    selected_files = np.random.choice(os.listdir(directory), size=num_images, replace=False)
    extra_images = []
    extra_labels = []

    for filename in selected_files:
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        img = img_to_array(img)
        img /= 255.0  # Normalize to [0, 1] if not already

        for _ in range(crops_per_image):
            crop = tf.image.random_crop(img, size=crop_shape)
            extra_images.append(crop.numpy())
            extra_labels.append([0]*(num_classes-1) + [1])  # Update according to your encoding

    return np.array(extra_images), np.array(extra_labels)

# Initialize variables for tracking the best model
best_val_accuracy = 0.0
best_epoch = 0

# Manually handle training and validation
for epoch in range(50):  # Assuming 50 epochs
    print(f"Epoch {epoch+1}/{50}")
    
    # Load and process extra images
    extra_images, extra_labels = load_and_process_images(directory='../../imagenet',
                                                         crop_shape=(32, 32, 3),
                                                         num_images=8,
                                                         crops_per_image=16)
    
    # Append extra images and labels to your training data
    X_train_augmented = np.concatenate((X_train, extra_images), axis=0)
    y_train_augmented = np.concatenate((y_train, extra_labels), axis=0)
    
    # Optionally shuffle the augmented dataset
    indices = np.arange(X_train_augmented.shape[0])
    np.random.shuffle(indices)
    X_train_augmented = X_train_augmented[indices]
    y_train_augmented = y_train_augmented[indices]
    
    # Train on the augmented dataset for this epoch
    history = model.fit(X_train_augmented, y_train_augmented,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        verbose=1)
    
    # Check if the validation accuracy of this epoch is the best so far
    val_accuracy = history.history['val_accuracy'][-1]
    if val_accuracy > best_val_accuracy:
        print(f"New best model found at epoch {epoch+1} with validation accuracy {val_accuracy}. Saving model.")
        model.save('models/best_model_so_far.h5')
        best_val_accuracy = val_accuracy
        best_epoch = epoch

    # You can also implement early stopping manually by checking if the current epoch - best_epoch exceeds your patience

# After training, you might want to save the final model as well
model.save('models/final_model.h5')


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model import create_model
from data_prep import X_train, y_train, X_val, y_val, X_test, y_test
import cv2
from tqdm import tqdm
from pathlib import Path

# Model parameters
input_shape = X_train.shape[1:]  # e.g., (32, 32, 3) for 32x32 RGB images
# Modify y_train to add an empty extra class
y_train = np.concatenate([y_train, np.zeros_like(y_train[:, :1])], axis=1)
y_val = np.concatenate([y_val, np.zeros_like(y_val[:, :1])], axis=1)
num_classes = y_train.shape[1]

model = create_model(input_shape, num_classes)

def load_and_process_images(directory, crop_shape, num_images=8, crops_per_image=16):
    selected_files = np.random.choice(os.listdir(directory), size=num_images, replace=False)
    extra_images = []
    extra_labels = []

    print("Loading negative ImageNet images")
    for filename in tqdm(selected_files):
        try:
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = img_to_array(img)
            img /= 255.0  # Normalize to [0, 1] if not already

            for _ in range(crops_per_image):
                crop = tf.image.random_crop(img, size=crop_shape)
                extra_images.append(crop.numpy())
                extra_labels.append([0]*(num_classes-1) + [1])
        except Exception as e:
            print(f"Error on image {filename}: {e}")

    return np.array(extra_images), np.array(extra_labels)

# Initialize variables for tracking the best model
best_val_accuracy = 0.0
best_epoch = 0

# Manually handle training and validation
for epoch in range(500):
    print(f"Epoch {epoch+1}/{50}")
    
    # Load and process extra images
    extra_images_all, extra_labels_all = load_and_process_images(directory='../../imagenet',
                                                         crop_shape=(32, 32, 3),
                                                         num_images=2048,
                                                         crops_per_image=16)
    extra_val_images, extra_val_labels = extra_images_all[:256], extra_labels_all[:256]
    extra_train_images, extra_train_labels = extra_images_all[256:], extra_labels_all[256:]

    # Append extra images and labels to your training data
    X_train_augmented = np.concatenate((X_train, extra_train_images), axis=0)
    y_train_augmented = np.concatenate((y_train, extra_train_labels), axis=0)
    X_val_augmented = np.concatenate((X_val, extra_val_images), axis=0)
    y_val_augmented = np.concatenate((y_val, extra_val_labels), axis=0)

    indices = np.arange(X_train_augmented.shape[0])
    np.random.shuffle(indices)
    X_train_augmented = X_train_augmented[indices]
    y_train_augmented = y_train_augmented[indices]

    def augment_image(image):
        image = tf.convert_to_tensor(image)
        image = tf.reverse(image, axis=[-1])
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.reverse(image, axis=[-1])
        return image

    X_train_augmented = augment_image(X_train_augmented)
    X_val_augmented = augment_image(X_val_augmented)

    save_dir = Path.home() / "imgs"
    save_dir.mkdir(exist_ok=True)
    for i, img in enumerate(X_train_augmented[:256]):
        # Construct a filename for each image
        filename = f"image_{i}.png"
        file_path = os.path.join(save_dir, filename)
        cv2.imwrite(file_path, (255 * img.numpy()).astype(np.uint8))
    
    # Train on the augmented dataset for this epoch
    history = model.fit(X_train_augmented, y_train_augmented,
                        batch_size=32,
                        validation_data=(X_val_augmented, y_val_augmented),
                        verbose=1)
    
    # Check if the validation accuracy of this epoch is the best so far
    val_accuracy = history.history['val_accuracy'][-1]
    if val_accuracy > best_val_accuracy:
        print(f"New best model found at epoch {epoch+1} with validation accuracy {val_accuracy}. Saving model.")
        model.save('models/best_model_so_far.h5')
        best_val_accuracy = val_accuracy
        best_epoch = epoch

    if epoch % 10 == 0:
        model.save(f'models/epoch_{epoch}.h5')

    # You can also implement early stopping manually by checking if the current epoch - best_epoch exceeds your patience

# After training, you might want to save the final model as well
model.save('models/final_model.h5')


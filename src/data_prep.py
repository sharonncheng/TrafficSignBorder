import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_image(image_path, image_size=(32, 32)):
    """Load an image from the path and preprocess it."""
    # debug
    print(f"Attempting to load image from path: {image_path}")
    image = cv2.imread(image_path)

    # show paths that cannot be loaded
    if image is None:
        print(f"Error: Could not load image at {image_path}.")
        return None
    image = cv2.resize(image, image_size) 

    # normalize pixel values to [0, 1]
    image = image / 255.0  
    return image

def prepare_data(csv_path, data_dir, image_size=(32, 32)):
    """Load and preprocess the dataset based on CSV files."""
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        image_path = os.path.join(data_dir, row['Path'])
        image = load_and_preprocess_image(image_path, image_size)
        # only append if the image was successfully loaded
        if image is not None:  
            images.append(image)
            labels.append(row['ClassId'])
    
    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=len(df['ClassId'].unique())) 
    
    return train_test_split(images, labels, test_size=0.2, random_state=42)

train_csv_path = '../data/raw/GTSRB/Train.csv'
test_csv_path = '../data/raw/GTSRB/Test.csv'
data_dir = '../data/raw/GTSRB'
train_data_dir = '../data/raw/GTSRB/Train'
test_data_dir = '../data/raw/GTSRB/Test'

# desired image size
image_size = (32, 32)  

X_train, X_val, y_train, y_val = prepare_data(train_csv_path, data_dir, image_size)
# Prepare testing data
# Note: For testing, you might want to keep a separate script or modify the function to not split data.
X_test, _, y_test, _ = prepare_data(test_csv_path, data_dir, image_size)

print(f'Training set size: {X_train.shape[0]}')
print(f'Validation set size: {X_val.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')

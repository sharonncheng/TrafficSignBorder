import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model
from data_prep import X_train, y_train, X_val, y_val, X_test, y_test

# Assuming X_train, X_val, y_train, y_val are loaded from your data preparation step
# For example:
# X_train, X_val, y_train, y_val = np.load('path/to/your/dataset.npy')

# Model parameters
input_shape = X_train.shape[1:]  # e.g., (32, 32, 3) for 32x32 RGB images
num_classes = y_train.shape[1]  # Based on your dataset

model = create_model(input_shape, num_classes)

# Callbacks
checkpoint = ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping])

# Save the final model
model.save('models/final_model.h5')

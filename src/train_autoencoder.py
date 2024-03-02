import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from data_prep import X_train

# Define the encoder part
def build_encoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(256, activation='relu')(x)
    return models.Model(inputs, encoded, name="encoder")

# Define the decoder part
def build_decoder(encoded_shape):
    encoded_input = tf.keras.Input(shape=encoded_shape)
    x = layers.Dense(4*4*128, activation='relu')(encoded_input)
    x = layers.Reshape((4, 4, 128))(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return models.Model(encoded_input, decoded, name="decoder")

# Build the autoencoder
input_shape = (32, 32, 3)
encoder = build_encoder(input_shape)
decoder = build_decoder(encoder.output_shape[1:])
autoencoder = models.Model(encoder.input, decoder(encoder.output))

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Summary of the model
autoencoder.summary()

# Assuming you have your dataset loaded in X_train (for training)
# X_train should be normalized to be between 0 and 1

checkpoint_cb = ModelCheckpoint(
    'autoencoder_{epoch:02d}.h5',
    save_freq='epoch',
    period=10,
    save_best_only=False,
    verbose=1,
)
autoencoder.fit(X_train, X_train, epochs=250, batch_size=256, shuffle=True, validation_split=0.2, callbacks=[checkpoint_cb])

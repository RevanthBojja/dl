import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1. Load ,normalize and reshape the MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 2. Add Gaussian noise to the images
# Adds random Gaussian noise to your clean training images to create corrupted images
# loc is mean and scale is spread of noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to be between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Displaying ten noisy images
n = 10
plt.figure(figsize=(20, 2)) #figure width 20 inches and height 2 inches
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)  #plot imges in 1 row 10 columns
    plt.imshow(x_test_noisy[i].reshape(28, 28))  #(28,28,1) image reshape to (28,28)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Build the autoencoder

input_img = tf.keras.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
print(x.shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
print(encoded.shape)

# At this point the representation is (7, 7, 32)
# Decoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
print(x.shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
print(x.shape)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Create Autoencoder Model
autoencoder = models.Model(input_img, decoded)
# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train the model
autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

decoded_imgs = autoencoder.predict(x_test_noisy)

# Show ten noisy and denoised images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Noisy Input
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Denoised Output
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
plt.show()

# ---------------- MAIN IDEA: Denoising Autoencoder ----------------

# This program implements a Convolutional Denoising Autoencoder.
# It learns to remove noise from images by reconstructing clean images
# from noisy inputs.

# ---------------- CORE CONCEPT ----------------

# Instead of just compressing data, this autoencoder learns:
# Noisy Image → Clean Image

# The model is trained by:
# - Input → noisy images
# - Target → original clean images

# This forces the network to learn meaningful features and ignore noise.

# ---------------- ARCHITECTURE DETAILS ----------------

# INPUT:
# Image → (28, 28, 1)

# ENCODER (Feature Compression):

# Conv2D(32, 3x3, same)
# Output → (28, 28, 32)

# MaxPooling(2x2)
# Output → (14, 14, 32)

# Conv2D(32, 3x3, same)
# Output → (14, 14, 32)

# MaxPooling(2x2)
# Output → (7, 7, 32)
# → This is the compressed latent representation

# DECODER (Reconstruction):

# Conv2D(32, 3x3, same)
# Output → (7, 7, 32)

# UpSampling(2x2)
# Output → (14, 14, 32)

# Conv2D(32, 3x3, same)
# Output → (14, 14, 32)

# UpSampling(2x2)
# Output → (28, 28, 32)

# Final Conv2D(1, 3x3, sigmoid)
# Output → (28, 28, 1)
# → Reconstructed clean image

# ---------------- EXECUTION OF EXPERIMENT ----------------

# 1. DATASET:
# MNIST handwritten digits

# 2. PREPROCESSING:
# - Normalize pixel values → [0,1]
# - Reshape → (28,28,1)

# 3. ADDING NOISE:
# Gaussian noise added:
# noisy_image = original + noise_factor * random_noise
# Values clipped between 0 and 1

# 4. TRAINING:
# - Input → noisy images
# - Output → clean images
# - Optimizer: Adam
# - Loss: Binary Crossentropy
# - Epochs: 10

# 5. LEARNING:
# Model learns to remove noise while preserving digit structure

# 6. OUTPUT:
# Shows comparison:
# Top row → noisy images
# Bottom row → denoised (reconstructed) images

# ---------------- SUMMARY ----------------

# Noisy Image → Encode (compress features) → Decode → Clean Image

# KEY IDEA:
# Learn robust features that ignore noise and retain important structure

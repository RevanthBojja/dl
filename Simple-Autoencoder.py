import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and flatten
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# Define size of encoding (compressed representation)
encoding_dim = 32  # smaller = more compression

# Input layer
input_img = Input(shape=(784,))

# Encoder
encoded = Dense(encoding_dim, activation='relu')(input_img)
print(encoded.shape)

# Decoder
decoded = Dense(784, activation='sigmoid')(encoded)
print(decoded.shape)

# Create Autoencoder model
autoencoder = Model(input_img, decoded)

# Compile
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
encoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()
# ---------------- MAIN IDEA: Autoencoder ----------------

# An Autoencoder is a neural network used for unsupervised learning.
# It learns to compress data into a smaller representation (encoding)
# and then reconstruct the original input from that compressed form.

# ---------------- CORE CONCEPT ----------------

# The network has two main parts:
# - Encoder → compresses input into a lower-dimensional representation
# - Decoder → reconstructs the original data from this compressed form

# The goal is to minimize the difference between input and output.

# ---------------- ARCHITECTURE DETAILS ----------------

# INPUT:
# Original MNIST image → (28, 28)
# Flattened → (784,)

# ENCODER:
# Dense(32)
# Output → (32,)
# → This is the compressed representation (latent space)

# DECODER:
# Dense(784)
# Output → (784,)
# → Reconstructed image

# FINAL MODEL:
# Input (784) → Encoded (32) → Decoded (784)

# ---------------- EXECUTION OF EXPERIMENT ----------------

# 1. DATASET:
# MNIST dataset (handwritten digits)

# 2. PREPROCESSING:
# - Normalize pixel values → [0,1]
# - Flatten images → (784,)

# 3. TRAINING:
# - Input = Output (x_train → x_train)
# - Optimizer: Adam
# - Loss: Binary Crossentropy
# - Epochs: 10
# - Batch size: 256

# 4. LEARNING:
# Model learns to retain important features while compressing data

# 5. PREDICTION:
# autoencoder.predict(x_test)
# → Outputs reconstructed images

# ---------------- SUMMARY ----------------

# Image → Compress (Encoder) → Latent Space → Reconstruct (Decoder)

# KEY IDEA:
# Learn efficient data representation by forcing compression

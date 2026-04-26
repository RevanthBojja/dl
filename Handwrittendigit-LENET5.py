# Step 1: Import libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Step 2: Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 3: Preprocessing
# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to (28,28,1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Step 4: Build LeNet-5 Model
model = keras.Sequential()

# C1: Conv Layer
model.add(layers.Conv2D(6, kernel_size=(5,5), activation='tanh', input_shape=(28,28,1)))

# S2: Pooling
model.add(layers.AveragePooling2D(pool_size=(2,2)))

# C3: Conv Layer
model.add(layers.Conv2D(16, kernel_size=(5,5), activation='tanh'))

# S4: Pooling
model.add(layers.AveragePooling2D(pool_size=(2,2)))

# Flatten
model.add(layers.Flatten())

# Fully Connected Layers
model.add(layers.Dense(120, activation='tanh'))
model.add(layers.Dense(84, activation='tanh'))

# Output Layer
model.add(layers.Dense(10, activation='softmax'))

# Step 5: Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train
history= model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# Step 7: Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

last_train_acc = history.history['accuracy'][-1]
print("Last Epoch Training Accuracy:", last_train_acc)

last_val_acc = history.history['val_accuracy'][-1]
print("Last Epoch Validation Accuracy:", last_val_acc)

print("Test Accuracy:", test_acc)

# sample image
import matplotlib.pyplot as plt
plt.imshow(x_train[0].reshape(28,28), cmap='gray')
plt.title("Sample Image")
plt.show()
# prediction of sample image
prediction = model.predict(x_train[0].reshape(1,28,28,1))
predicted_label = np.argmax(prediction)
print("\nPredicted Label:", predicted_label)

# ---------------- MAIN IDEA: LeNet-5 ----------------

# LeNet-5 is a Convolutional Neural Network (CNN) designed for image classification,
# especially handwritten digit recognition (like MNIST).

# CORE CONCEPT:
# Instead of manually defining features, LeNet-5 automatically learns patterns
# from images using convolution filters.

# WHAT IT DOES:
# - Detects low-level features (edges, lines) in early layers
# - Builds higher-level features (shapes, digit structures) in deeper layers
# - Uses fully connected layers to classify the image into one of the 10 digits (0–9)

# In short:
# Image → Feature Extraction (Conv + Pool) → Pattern Learning → Classification
# ---------------- LE-Net5 on MNIST — CRISP DESCRIPTION ----------------

# DATASET:
# MNIST → 60,000 training images + 10,000 test images
# Each image: 28x28 grayscale → shape: (28, 28)

# AFTER LOADING:
# x_train shape → (60000, 28, 28)
# x_test  shape → (10000, 28, 28)
# y_train shape → (60000,)
# y_test  shape → (10000,)

# PREPROCESSING:

# 1. NORMALIZATION:
# Pixel values scaled from [0,255] → [0,1]

# 2. RESHAPING:
# CNN expects channel dimension → (height, width, channels)
# x_train → (60000, 28, 28, 1)
# x_test  → (10000, 28, 28, 1)

# 3. ONE-HOT ENCODING:
# Labels converted to vectors of size 10
# Example: digit 3 → [0 0 0 1 0 0 0 0 0 0]
# y_train → (60000, 10)
# y_test  → (10000, 10)

# ---------------- MODEL ARCHITECTURE (LeNet-5) ----------------

# INPUT:
# (28, 28, 1)

# C1: Conv2D (6 filters, 5x5, stride=1, valid padding)
# Output → (24, 24, 6)
# Formula: (28 - 5 + 1 = 24)

# S2: AveragePooling (2x2)
# Output → (12, 12, 6)

# C3: Conv2D (16 filters, 5x5)
# Output → (8, 8, 16)
# Formula: (12 - 5 + 1 = 8)

# S4: AveragePooling (2x2)
# Output → (4, 4, 16)

# FLATTEN:
# (4 × 4 × 16) = 256 → shape: (256,)

# FULLY CONNECTED LAYERS:

# Dense (120 neurons)
# Output → (120,)

# Dense (84 neurons)
# Output → (84,)

# OUTPUT LAYER:
# Dense (10 neurons, softmax)
# Output → (10,) → probability distribution over digits (0–9)

# ---------------- TRAINING ----------------

# LOSS FUNCTION:
# categorical_crossentropy → for multi-class classification

# OPTIMIZER:
# Adam → adaptive learning rate

# BATCH SIZE:
# 64 → number of samples per update

# EPOCHS:
# 5 → full passes through dataset

# VALIDATION SPLIT:
# 10% of training data used for validation

# ---------------- OUTPUT METRICS ----------------

# Training Accuracy:
# Accuracy on training data (last epoch)

# Validation Accuracy:
# Accuracy on unseen validation data

# Test Accuracy:
# Final evaluation on test dataset (10000 images)

# ---------------- PREDICTION ----------------

# Input shape for prediction:
# (1, 28, 28, 1)

# Model outputs:
# Array of 10 probabilities

# np.argmax():
# Returns index of highest probability → predicted digit

# ---------------- FLOW SUMMARY ----------------

# Image (28x28) →
# Conv → Pool → Conv → Pool →
# Flatten →
# Dense → Dense →
# Softmax →
# Digit Prediction (0–9)

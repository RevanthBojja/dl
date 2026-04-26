# Import required libraries
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D

# ---------------- STEP 1: LOAD AND DISPLAY ORIGINAL IMAGE ----------------
image_path = "/content/jayaramphoto.JPG"

image = load_img(image_path)  # Load image in RGB format
plt.imshow(image)             # Display image
plt.axis("off")               # Hide axes
plt.show()

image_array = img_to_array(image)  # Convert image to NumPy array
print(image_array.shape)           # (height, width, channels)
print(image_array[0])              # First row pixel values

# ---------------- STEP 2: RESIZE IMAGE ----------------
image_resized = load_img(image_path, target_size=(224, 224))  # Resize to 224x224
image_array = img_to_array(image_resized)
image_shape = image_array.shape
print(image_shape)

plt.imshow(image_resized)  # Display resized image
plt.axis("off")
plt.show()

# ---------------- STEP 3: APPLY RANDOM CONVOLUTION FILTER ----------------
model = Sequential([
    Conv2D(filters=1, kernel_size=(3,3), strides=(1,1),
           activation=None, input_shape=image_shape)
])

# Apply convolution (add batch dimension)
filtered_image = model.predict(image_array.reshape(1, 224, 224, 3))

plt.imshow(filtered_image[0])  # Display filtered image
plt.axis("off")
plt.show()

# Get weights of the first convolutional layer
conv_layer = model.layers[0]
weights, biases = conv_layer.get_weights()

print("Kernel Weights Shape:", weights.shape)
print(weights)

# ---------------- STEP 4: CONVERT IMAGE TO GRAYSCALE ----------------
gray_image = load_img(image_path, color_mode="grayscale")
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.show()

image_array = img_to_array(gray_image)
print(image_array.shape)  # (height, width, 1)

gray_image = load_img(image_path, color_mode="grayscale", target_size=(224, 224))
image_array = img_to_array(gray_image)
image_shape = image_array.shape
print(image_shape)

# ---------------- STEP 5: APPLY CUSTOM KERNEL ----------------
# Define a custom 3x3 kernel
kernel = np.array([[[-1, 1, 1],
                    [-1, 1, 1],
                    [-1, 1, 1]]], dtype=np.float32)

# Reshape to match Conv2D weight format (height, width, input_channels, output_channels)
kernel = kernel.reshape((3, 3, 1, 1))

# Create a new CNN model for grayscale input
model = Sequential([
    Conv2D(filters=1, kernel_size=(3,3), strides=(1,1),
           activation=None, input_shape=image_shape)
])

# Set the weights of the convolutional layer to your custom kernel
bias = np.zeros(1)
model.layers[0].set_weights([kernel, bias])

# Apply the convolution filter
filtered_image = model.predict(image_array.reshape(1, 224, 224, 1))

plt.imshow(filtered_image[0], cmap="gray")  # Display result
plt.axis("off")
plt.show()


# ---------------- FULL DESCRIPTION BELOW (AS COMMENTS) ----------------

# This entire script demonstrates how Convolutional Neural Networks (CNNs)
# process images step by step.

# 1. IMAGE LOADING:
# The image is loaded using Keras utility functions and displayed using matplotlib.
# It is then converted into a NumPy array because neural networks require numerical input.

# 2. IMAGE RESIZING:
# The image is resized to 224x224, which is a common input size for CNN models.
# This ensures consistent dimensions when feeding data into the network.

# 3. RANDOM CONVOLUTION:
# A Conv2D layer is created with a randomly initialized 3x3 filter.
# When applied, this filter slides over the image (convolution operation)
# and produces a transformed output highlighting certain patterns.

# 4. KERNEL INSPECTION:
# The weights (kernel values) of the convolution layer are extracted.
# These values define how the filter detects features in the image.

# 5. GRAYSCALE CONVERSION:
# The image is converted from RGB (3 channels) to grayscale (1 channel).
# This simplifies the data and reduces computational complexity.

# 6. CUSTOM KERNEL APPLICATION:
# A custom kernel is manually defined.
# This kernel is designed to highlight vertical edges in the image.

# The kernel is reshaped into (3,3,1,1) format:
# (height, width, input_channels, number_of_filters)

# The model's Conv2D layer weights are replaced with this custom kernel.

# When convolution is applied again, the output image clearly shows
# edge-detected features based on the custom filter.

# FINAL RESULT:
# You see how:
# - Random filters produce arbitrary transformations
# - Custom filters can be designed to extract specific features (like edges)

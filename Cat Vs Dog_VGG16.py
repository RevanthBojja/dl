import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogsvscats
import zipfile
zip_ref = zipfile.ZipFile('/content/dogsvscats.zip','r')
zip_ref.extractall('/content')
zip_ref.close()
# Generators
# class 0 indicates cat and class 1 indicates dog
train_ds = keras.utils.image_dataset_from_directory(
    directory ='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
)
test_ds = keras.utils.image_dataset_from_directory(
    directory ='/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256),
)
# Normalize
def process(image,label):
  image = tf.cast(image/255. ,tf.float32)
  return image,label
train_ds = train_ds.map(process)
test_ds = test_ds.map(process)
# Load Pre-trained VGG16 Model (without top classification layers)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

# Freeze pre-trained layers to stop retain learned features
base_model.trainable = False

# Add Custom Layers
x = Flatten()(base_model.output)
print(x.shape)
x = Dense(256, activation="relu")(x)
print(x.shape)
x = Dropout(0.5)(x)
print(x.shape)
x = Dense(1, activation="sigmoid")(x)  # Return the probabolity of Dog class
print(x.shape)

# Create new model
model = Model(inputs=base_model.input, outputs=x)
# Model summary
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_ds,epochs=10,validation_data=test_ds)
print("Training Accuracy:",history.history['accuracy'][-1]*100)
print("Validation Accuracy: ", history.history['val_accuracy'][-1]*100)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()
import cv2
test_img = cv2.imread('/content/test/dogs/dog.10006.jpg')
plt.imshow(test_img)
test_img.shape
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
p=model.predict(test_input)
print(p)
# Return probablity is greaterthan or equal then ouput is class 1 else output is class 0.
if(p>=0.5):
 print("DOG")
else:
 print("CAT")
test_img = cv2.imread('/content/test/cats/cat.10030.jpg')
plt.imshow(test_img)
test_img.shape
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
p=model.predict(test_input)
print(p)
# Return probablity is greaterthan or equal then ouput is class 1 else output is class 0.
if(p>=0.5):
  print("DOG")
else:
  print("CAT")

# ---------------- MAIN IDEA: VGG16 (Transfer Learning) ----------------

# VGG16 is a deep Convolutional Neural Network (CNN) known for its simple and uniform architecture.
# It uses multiple stacked 3x3 convolution layers to learn rich image features.
# In this program, VGG16 is used as a pre-trained feature extractor (Transfer Learning)
# for Dog vs Cat classification.

# ---------------- CORE CONCEPT ----------------

# Instead of training from scratch, we reuse knowledge learned from ImageNet:
# - Early layers → detect edges, textures
# - Deeper layers → detect complex patterns (faces, shapes)
# We freeze these layers and only train new custom layers for our task.

# ---------------- ARCHITECTURE DETAILS ----------------

# INPUT:
# Image size → (256, 256, 3)

# BASE MODEL (VGG16 without top layers):
# Structure:
# Block1 → Conv(64) + Conv(64) + MaxPool → (128,128,64)
# Block2 → Conv(128) + Conv(128) + MaxPool → (64,64,128)
# Block3 → Conv(256) + Conv(256) + Conv(256) + MaxPool → (32,32,256)
# Block4 → Conv(512) + Conv(512) + Conv(512) + MaxPool → (16,16,512)
# Block5 → Conv(512) + Conv(512) + Conv(512) + MaxPool → (8,8,512)

# Final Output from VGG16 base:
# (8, 8, 512)

# FREEZING:
# base_model.trainable = False
# → Prevents updating pre-trained weights

# ---------------- CUSTOM CLASSIFIER ----------------

# Flatten:
# (8 × 8 × 512) = 32768

# Dense Layer:
# 32768 → 256 (ReLU activation)

# Dropout:
# 50% neurons dropped to reduce overfitting

# Output Layer:
# Dense(1, sigmoid)
# → Output: probability of Dog (class 1)

# ---------------- EXECUTION OF EXPERIMENT ----------------

# 1. DATASET:
# Dogs vs Cats dataset from Kaggle
# - Training: ~20000 images
# - Testing: ~5000 images

# 2. PREPROCESSING:
# - Resize images to (256,256)
# - Normalize pixel values to [0,1]

# 3. TRAINING:
# - Optimizer: Adam
# - Loss: Binary Crossentropy
# - Epochs: 10
# - Only custom layers are trained (feature extraction mode)

# 4. PERFORMANCE TRACKING:
# - Accuracy and loss plotted for training vs validation
# - Helps visualize learning and overfitting

# 5. PREDICTION:
# Input shape → (1, 256, 256, 3)
# Model outputs probability:
#   ≥ 0.5 → Dog
#   < 0.5 → Cat

# ---------------- SUMMARY ----------------

# Image → Pretrained Feature Extraction (VGG16) → Custom Dense Layers → Binary Classification

# Key Advantage:
# Faster training + better accuracy due to transfer learning

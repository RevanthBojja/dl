import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
train_ds = keras.utils.image_dataset_from_directory(
    directory ='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224),
)
test_ds = keras.utils.image_dataset_from_directory(
    directory ='/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224),
)
# 20000 training images and 10000 dogs and 10000 cats
# 5000 testing images and 2500 dogs and 2500 cats
# Normalize
def process(image,label):
  image = tf.cast(image/255. ,tf.float32)
  return image,label
train_ds = train_ds.map(process)
test_ds = test_ds.map(process)
# Build AlexNet model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

# 2nd Convolutional Layer
model.add(Conv2D(256, (5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

# 3rd Convolutional Layer
model.add(Conv2D(384, (3,3), padding='same', activation='relu'))

# 4th Convolutional Layer
model.add(Conv2D(384, (3,3), padding='same', activation='relu'))

# 5th Convolutional Layer
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

# Flatten
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Output Layer (Binary)
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
history=model.fit(train_ds,epochs=10,validation_data=test_ds,verbose=1)
print("Last Epoch Training Accuracy:", history.history['accuracy'][-1])
print("Last Epoch Validation Accuracy:", history.history['val_accuracy'][-1])
# Prediction of the image

import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('cat_or_dog.jpg', target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
import cv2
test_img = cv2.imread('/content/test/dogs/dog.10006.jpg')
plt.imshow(test_img)
print("test image shape:",test_img.shape)

test_img = cv2.resize(test_img,(224,224))
test_input = test_img.reshape((1,224,224,3))
p=model.predict(test_input) #sigmoid returns probability of class 1
if(p>=0.5):
 print("DOG")
else
 print("CAT")
test_img = cv2.imread('/content/test/cats/cat.10030.jpg')
plt.imshow(test_img)
print("test image shape:",test_img.shape)
test_img = cv2.resize(test_img,(224,224))
test_input = test_img.reshape((1,224,224,3))
p=model.predict(test_input)
if(p>=0.5):
  print("DOG")
else
  print("CAT")

# ---------------- MAIN IDEA: AlexNet ----------------

# AlexNet is a deep Convolutional Neural Network (CNN) used for image classification.
# In this program, it is applied to classify images as either Dog or Cat.

# ---------------- CORE CONCEPT ----------------

# AlexNet automatically learns hierarchical features from images:
# - Early layers → edges, colors, textures
# - Deeper layers → shapes, object parts (ears, face, etc.)
# - Final layers → decision (Dog vs Cat)

# ---------------- ARCHITECTURE DETAILS ----------------

# INPUT:
# Image size → (224, 224, 3)

# 1st Conv Layer:
# Conv2D(96 filters, 11x11, stride=4)
# Output → (54, 54, 96)
# MaxPooling(3x3, stride=2)
# Output → (26, 26, 96)

# 2nd Conv Layer:
# Conv2D(256 filters, 5x5, padding='same')
# Output → (26, 26, 256)
# MaxPooling(3x3, stride=2)
# Output → (12, 12, 256)

# 3rd Conv Layer:
# Conv2D(384 filters, 3x3, same)
# Output → (12, 12, 384)

# 4th Conv Layer:
# Conv2D(384 filters, 3x3, same)
# Output → (12, 12, 384)

# 5th Conv Layer:
# Conv2D(256 filters, 3x3, same)
# Output → (12, 12, 256)
# MaxPooling(3x3, stride=2)
# Output → (5, 5, 256)

# Flatten:
# (5 × 5 × 256) = 6400

# Fully Connected Layers:
# Dense(4096) → Dropout(0.5)
# Dense(4096) → Dropout(0.5)

# Output Layer:
# Dense(1, sigmoid) → outputs probability of "Dog"

# ---------------- EXECUTION OF EXPERIMENT ----------------

# 1. DATA LOADING:
# Dataset (Dogs vs Cats) downloaded from Kaggle and extracted

# 2. DATA PIPELINE:
# Images loaded using image_dataset_from_directory
# - Training set: 20000 images (10000 dogs + 10000 cats)
# - Test set: 5000 images (2500 dogs + 2500 cats)

# 3. PREPROCESSING:
# - Images resized to (224,224)
# - Pixel values normalized to [0,1]

# 4. TRAINING:
# - Optimizer: Adam
# - Loss: Binary Crossentropy
# - Epochs: 10
# - Model learns to distinguish features between dogs and cats

# 5. EVALUATION:
# - Training accuracy and validation accuracy printed after training

# 6. PREDICTION:
# - Input image reshaped to (1,224,224,3)
# - Model outputs probability:
#     > 0.5 → Dog
#     ≤ 0.5 → Cat

# ---------------- SUMMARY ----------------

# AlexNet performs:
# Image → Deep Feature Extraction → Dense Learning → Binary Classification (Dog/Cat)

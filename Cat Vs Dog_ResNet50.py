import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
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
# Load Pre-trained ResNet50 Model (without top classification layers)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256, 256, 3))

# Freeze pre-trained layers to stop retain learned features
base_model.trainable = False

# Add Custom Layers
x = Flatten()(base_model.output)
print(x.shape)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
# x= Dense(128, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)  # output is the probability of class1

# Create new model
model = Model(inputs=base_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_ds,epochs=5,validation_data=test_ds)
print("Training Accuracy:", history.history['accuracy'][-1]*100)
print("validation accuracy: ",history.history['val_accuracy'][-1]*100)
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
if(p>=0.5):
 print(p,":DOG")
else:
 print(p,":CAT")
test_img = cv2.imread('/content/test/cats/cat.10030.jpg')
plt.imshow(test_img)
test_img.shape
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
p=model.predict(test_input)
if(p>=0.5):
  print(p,":DOG")
else:
  print(p,":CAT")

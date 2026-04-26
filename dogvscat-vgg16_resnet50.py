# ================== IMPORTS ==================
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogsvscats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogsvscats.zip','r')
zip_ref.extractall('/content')
zip_ref.close()



# ================== DATASET ==================
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

# Normalize
def process(x, y):
    x = tf.cast(x / 255., tf.float32)
    return x, y

train_ds = train_ds.map(process)
test_ds  = test_ds.map(process)

# ================== MODEL ==================
input_layer = Input(shape=(256, 256, 3))

# Load pretrained models
vgg = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)

# Freeze layers
for layer in vgg.layers:
    layer.trainable = False

for layer in resnet.layers:
    layer.trainable = False

# Feature extraction
vgg_features = GlobalAveragePooling2D()(vgg.output)
resnet_features = GlobalAveragePooling2D()(resnet.output)

# Merge features
merged = Concatenate()([vgg_features, resnet_features])

# Classifier
x = Dense(256, activation='relu')(merged)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Final model
model = Model(inputs=input_layer, outputs=output)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ================== TRAIN ==================
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    verbose=1
)

# ================== RESULTS ==================
print("Training Accuracy:", history.history['accuracy'][-1]*100)
print("Validation Accuracy:", history.history['val_accuracy'][-1]*100)

```python id="hybrid_vgg_resnet_description"
# ---------------- MAIN IDEA: Hybrid VGG16 + ResNet50 ----------------

# This model combines two powerful pretrained CNNs — VGG16 and ResNet50 —
# to improve feature extraction for Dog vs Cat classification.

# ---------------- CORE CONCEPT ----------------

# Instead of relying on a single network, we use BOTH:
# - VGG16 → strong at capturing spatial patterns (simple, deep stacked convs)
# - ResNet50 → strong at learning deeper representations using residual connections

# Their features are merged to create a richer and more informative representation.

# ---------------- ARCHITECTURE DETAILS ----------------

# INPUT:
# (256, 256, 3)

# VGG16 OUTPUT:
# Final feature map → (8, 8, 512)
# After GlobalAveragePooling → (512,)

# ResNet50 OUTPUT:
# Final feature map → (8, 8, 2048)
# After GlobalAveragePooling → (2048,)

# FEATURE MERGING:
# Concatenate → (512 + 2048) = (2560,)

# CLASSIFIER:
# Dense(256) → (256,)
# Dense(128) → (128,)
# Dense(1, sigmoid) → (1,) → probability of Dog

# ---------------- FREEZING ----------------

# All layers in VGG16 and ResNet50 are frozen:
# → No weight updates
# → Only classifier layers are trained
# → Faster training + prevents overfitting

# ---------------- EXECUTION OF EXPERIMENT ----------------

# 1. DATASET:
# Dogs vs Cats dataset (Kaggle)
# Images resized to (256,256)

# 2. PREPROCESSING:
# Pixel normalization → [0,1]

# 3. TRAINING:
# - Optimizer: Adam
# - Loss: Binary Crossentropy
# - Epochs: 10
# - Only top dense layers are trained

# 4. FEATURE FLOW:
# Image →
#   → VGG16 → (512,)
#   → ResNet50 → (2048,)
#   → Merge → (2560,)
#   → Dense layers → Classification

# ---------------- SUMMARY ----------------

# Image → Dual Feature Extraction (VGG + ResNet) → Feature Fusion → Classification

# KEY ADVANTAGE:
# Combining two architectures improves feature diversity → better accuracy
```

# Chest-X-Ray-Pneumonia
# ===============================
# 1. Import Required Libraries
# ===============================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from IPython.display import display

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===============================
# 2. Set Random Seeds
# ===============================
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# 3. Load Dataset
# ===============================
# ✅ Change this path only if you renamed or re-uploaded
DATA_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray"
CATEGORIES = ["PNEUMONIA", "NORMAL"]
IMG_SIZE = 150

def load_images_from_folder(folder, label):
    data = []
    for filename in tqdm(os.listdir(folder), desc=f"Loading {label}"):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append([img, label])
        except:
            pass
    return data

# Load train, val, test data
train_data, val_data, test_data = [], [], []

for category in CATEGORIES:
    train_path = os.path.join(DATA_DIR, "train", category)
    val_path = os.path.join(DATA_DIR, "val", category)
    test_path = os.path.join(DATA_DIR, "test", category)

    train_data += load_images_from_folder(train_path, category)
    val_data += load_images_from_folder(val_path, category)
    test_data += load_images_from_folder(test_path, category)

# Shuffle datasets
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

def split_data(data):
    X = np.array([item[0] for item in data])
    y = np.array([item[1] for item in data])
    return X, y

X_train, y_train = split_data(train_data)
X_val, y_val = split_data(val_data)
X_test, y_test = split_data(test_data)

# Normalize and encode labels
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train), num_classes=2)
y_val_enc = to_categorical(le.transform(y_val), num_classes=2)
y_test_enc = to_categorical(le.transform(y_test), num_classes=2)

# ===============================
# 4. Data Augmentation
# ===============================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# ===============================
# 5. Build Model
# ===============================
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# ===============================
# 6. Compile Model
# ===============================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 7. Train Model
# ===============================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3)

history = model.fit(
    datagen.flow(X_train, y_train_enc, batch_size=32),
    validation_data=(X_val, y_val_enc),
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# ===============================
# 8. Evaluate Model
# ===============================
loss, accuracy = model.evaluate(X_test, y_test_enc)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# ===============================
# 9. Classification Report
# ===============================
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_enc, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=CATEGORIES))

# ===============================
# 10. Confusion Matrix
# ===============================
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

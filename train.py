import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

IMG_SIZE = (128, 128)
MODEL_PATH = "model.h5"

def ela_transform(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    resaved_path = "temp_ela.jpg"
    original.save(resaved_path, 'JPEG', quality=quality)
    resaved = Image.open(resaved_path)
    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(diff).enhance(scale)
    return ela_image

def load_ela_dataset(data_dir):
    X, y = [], []
    for label, folder in enumerate(["real", "fake"]):  # real=0, fake=1
        path = os.path.join(data_dir, folder)
        for file in os.listdir(path):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(path, file)
                ela_image = ela_transform(image_path).resize(IMG_SIZE)
                X.append(img_to_array(ela_image) / 255.0)
                y.append(label)
    return np.array(X), np.array(y)

print("[INFO] Loading dataset...")
X, y = load_ela_dataset("dataset")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("[INFO] Building model...")
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("[INFO] Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=30, batch_size=32, class_weight=class_weights)

model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")

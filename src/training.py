import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_data

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_dir, "data", "flowers")

# Load data
train_ds, val_ds = load_data(data_dir)

# Get class names before prefetching
num_classes = len(train_ds.class_names)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

def train_model():
    data_dir = r"c:\Users\peter\Downloads\More projects\NN\flower-recognition\data\flowers"
    train_ds, val_ds = load_data(data_dir)
    num_classes = len(train_ds.class_names)
    input_shape = (224, 224, 3)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save('flower_model.h5')
    print("Model saved as flower_model.h5")

if __name__ == "__main__":
    train_model()

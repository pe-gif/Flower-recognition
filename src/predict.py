import tensorflow as tf
from PIL import Image
import numpy as np
import sys
import os

# Load the trained model
model = tf.keras.models.load_model('flower_model.h5')

# Get class names from the training set directory
data_dir = r"C:\Users\peter\Downloads\flowers"
class_names = sorted(os.listdir(data_dir))

# Get image path from command line
image_path = sys.argv[1]

# Preprocess the image
img = Image.open(image_path).resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

print(f"Predicted flower: {class_names[predicted_class]}")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model("fruit_model.h5")

# Class labels - use the same folder names you used during training
class_names = ['apple', 'banana', 'grapes', 'mango', 'orange']

# Load an image to predict
img_path = 'test/apple1.jpg'  # path to your test image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted class: {predicted_class}")

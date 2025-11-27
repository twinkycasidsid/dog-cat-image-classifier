import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Constants
TRAIN_DIR = 'train'
IMAGE_CHANNELS = 3
MODEL_NAME = 'cat_dog_classifier.h5'
TEST_IMG = 'dog9.jpg'

# Load model
model = load_model(MODEL_NAME)

# Get expected input size
input_shape = model.input_shape[1:3]  # Exclude batch size

# Load and preprocess the test image
test_img = cv2.imread(TEST_IMG)
if test_img is None:
    print(f"Error loading test image: {TEST_IMG}")
    exit()

original_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, input_shape)  # Resize to match model input
test_img = test_img / 255.0
data = test_img.reshape(1, *input_shape, IMAGE_CHANNELS)

# Predict the class
predictions = model.predict(data)[0]
all_classes = ['cats', 'dogs']  # Adjust according to your dataset
for i, prob in enumerate(predictions):
    print(f"{all_classes[i]}: {prob * 100:.2f}%")

predicted_class = all_classes[np.argmax(predictions)]
accuracy_percentage = np.max(predictions) * 100

# Annotate and display the image
plt.imshow(original_img)
plt.title(f"Predicted: {predicted_class} \nAccuracy: {accuracy_percentage:.2f}%")
plt.axis('off')
plt.show()

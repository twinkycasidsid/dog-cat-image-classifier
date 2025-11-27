import os
import numpy as np
import cv2
from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Directories and hyperparameters
TRAIN_DIR = 'train'  # Ensure the directory contains subfolders: 'cat' and 'dog'
IMG_SIZE = 128
IMAGE_CHANNELS = 3
FIRST_NUM_CHANNEL = 32
FILTER_SIZE = 3
LR = 0.0001
PERCENT_TRAINING_DATA = 80
NUM_EPOCHS = 50
MODEL_NAME = 'cat_dog_classifier.h5'

# Function to define classes (cat and dog)
def define_classes():
    all_classes = [folder for folder in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, folder))]
    return all_classes, len(all_classes)

# Function to load and preprocess images
def create_train_data(all_classes):
    training_data = []
    for label_index, specific_class in enumerate(all_classes):
        current_dir = os.path.join(TRAIN_DIR, specific_class)
        print(f'Reading directory of {current_dir}')
        for img_filename in os.listdir(current_dir):
            path = os.path.join(current_dir, img_filename)
            img = cv2.imread(path)
            if img is None:
                print(f"Skipping invalid image: {path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize image
            training_data.append([img, label_index])
    shuffle(training_data)
    return training_data

# Prepare data
all_classes, NUM_OUTPUT = define_classes()  # e.g., ['cat', 'dog']
print(f"Classes found: {all_classes}")
training_data = create_train_data(all_classes)

# Split into training and test sets
split_index = int(len(training_data) * (PERCENT_TRAINING_DATA / 100))
train = training_data[:split_index]
test = training_data[split_index:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
Y_train = to_categorical([i[1] for i in train], NUM_OUTPUT)
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)
Y_test = to_categorical([i[1] for i in test], NUM_OUTPUT)

# Build the model
model = Sequential([
    Conv2D(FIRST_NUM_CHANNEL, (FILTER_SIZE, FILTER_SIZE), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, IMAGE_CHANNELS)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 2, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 4, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(FIRST_NUM_CHANNEL * 8, (FILTER_SIZE, FILTER_SIZE), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(FIRST_NUM_CHANNEL * 16, activation='relu'),
    Dropout(0.5),
    Dense(NUM_OUTPUT, activation='softmax')  # Outputs 2 classes: cat and dog
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=16, validation_data=(X_test, Y_test), verbose=1)

# Save the model
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")

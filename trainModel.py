import os
import random
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_dir = "data_path"
class_names = os.listdir(data_dir)

images = []
labels = []

image_size = (224, 224)

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for image_filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_filename)
        image = cv2.imread(image_path)
        print(f"Loaded image: {image_path}")
        image = cv2.resize(image, image_size)
        images.append(image)

        label = 1 if class_name == 'dog' else 0
        labels.append(label)

X = np.array(images)
Y = np.array(labels)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),  # Adjust input shape based on your image size
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (dog or not dog)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 10

print("Training started...")
history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val))

print("Training completed.")

save_path = "save_path"
model.save(save_path)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

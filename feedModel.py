import tensorflow as tf
import cv2
import numpy as np

# Load model
saved_model_path = "model_path"
model = tf.keras.models.load_model(saved_model_path)

# Load and preprocess the input image
input_image_path = "image_path"
input_image = cv2.imread(input_image_path)
input_image = cv2.resize(input_image, (224, 224))  # Resize to the model's input size
input_image = input_image / 255.0  # Normalize pixel values

# Reshape the image if needed (e.g., for a single image)
input_image = np.expand_dims(input_image, axis=0)

# Make predictions
predictions = model.predict(input_image)

# Interpret predictions
if predictions[0][0] >= 0.5:
    result = "It's a dog!"
else:
    result = "It's not a dog."

print(result)

# Display the image and result
cv2.imshow("Image", input_image[0])  # Show the input image
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window

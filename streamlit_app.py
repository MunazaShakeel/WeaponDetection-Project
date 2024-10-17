import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load your trained model
model_path = '/workspaces/WeaponDetection-Project/cnn_model1.h5'  # Update this path to where your model is saved
model = tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size=(60, 60)):
    """Loads and preprocesses the image."""
    # Convert the uploaded image to a NumPy array
    img = np.array(image)

    # Resize the image
    img_resized = cv2.resize(img, target_size)

    # Convert to grayscale (if using grayscale)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Normalize pixel values (0-1 range)
    img_normalized = img_gray / 255.0

    # Reshape for the model input (assuming your model expects grayscale input)
    img_reshaped = img_normalized.reshape((60, 60, 1))  # Shape (160, 120, 1)

    # Add batch dimension
    return np.array([img_reshaped])  # Shape (1, 160, 120, 1)

def predict_weapon(image):
    """Predicts if a weapon is present in the image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Define class labels (adjust based on your model's output)
    class_labels = ['Guns', 'Knives', 'NoWeapons']
    predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability
    return class_labels[predicted_class]  # Return the class label

# Streamlit app interface
st.title("Weapon Detection App")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict if a weapon is present in the uploaded image
    result = predict_weapon(image)

    # Output the prediction
    st.write(f"**Prediction:** {result}")

    if result == 'Weapon':
        st.write("⚠️ A weapon was detected!")
    else:
        st.write("✅ No weapon detected.")

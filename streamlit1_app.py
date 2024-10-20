import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# Path to your model file
model = 'cnn_model1.h5'

# Check if the model exists and load it
if os.path.exists(model):
    model = tf.keras.models.load_model(model)
    # Compile the model with an appropriate loss, optimizer, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.write("Model loaded and compiled successfully.")
else:
    st.write(f"Model not found at {model}. Please check the path.")

# Preprocessing function
def preprocess_image(image, target_size=(60, 60)):
    """Preprocess the uploaded image to the format required by the model."""
    img = np.array(image)
    img_resized = cv2.resize(img, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    img_normalized = img_gray / 255.0
    img_reshaped = img_normalized.reshape((60, 60, 1))
    return np.array([img_reshaped])  # Add batch dimension

# Prediction function
def predict_weapon(image):
    """Predicts if a weapon is present in the image and applies a confidence threshold."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Define class labels
    class_labels = ['Guns', 'Knives', 'NoWeapons']

    # Get prediction probabilities
    predicted_probs = prediction[0]  # First element in prediction array

    # Set a confidence threshold
    confidence_threshold = 0.5

    # Get the index of the class with the highest probability
    predicted_class_idx = np.argmax(predicted_probs)

    # If the highest confidence is below the threshold, classify as "NoWeapons"
    if predicted_probs[predicted_class_idx] < confidence_threshold:
        return "NoWeapons", predicted_probs

    # Return the predicted class and probabilities
    return class_labels[predicted_class_idx], predicted_probs

# Streamlit app interface
st.title("Weapon Detection App")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict if a weapon is present in the uploaded image
    result, confidence = predict_weapon(image)

    # Output the prediction and confidence scores
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence Scores:** {confidence}")

    if result == 'Guns':
        st.write("⚠️ A gun was detected with high confidence!")
    elif result == 'Knives':
        st.write("⚠️ A knife was detected!")
    else:
        st.write("No weapon detected.")

# Output TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

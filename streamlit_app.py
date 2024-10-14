# prompt: so according to this give me the streamlit code

import streamlit as st
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

# Load the trained SVM model
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Load the trained CNN model
cnn_model = load_model('cnn_model.h5')

# Function to extract HOG features
def extract_hog_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Function to preprocess the image for CNN
def preprocess_image_cnn(image_path, target_size=(60, 60)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    img_array = img_normalized.reshape(1, 60, 60, 1)
    return img_array

# Label encoder (you might need to define this based on how you trained your models)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Guns', 'Knives', 'NoWeapons'])  # Replace with your actual classes

# Streamlit app
st.title("Weapon Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image for both SVM and CNN
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (90, 90))
    hog_features = extract_hog_features(resized_img)
    hog_features_flattened = hog_features.flatten().reshape(1, -1)

    img_array_cnn = preprocess_image_cnn(uploaded_file)

    # Predict using SVM
    svm_prediction = svm_model.predict(hog_features_flattened)
    svm_predicted_label = label_encoder.inverse_transform(svm_prediction)[0]

    # Predict using CNN
    cnn_prediction = cnn_model.predict(img_array_cnn)
    cnn_predicted_label = label_encoder.inverse_transform([np.argmax(cnn_prediction)])[0]

    st.write("SVM Prediction:", svm_predicted_label)
    st.write("CNN Prediction:", cnn_predicted_label)
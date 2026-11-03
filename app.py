import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Melanoma Skin Cancer Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("melanoma_cnn_model.keras")

model = load_model()
class_names = ["Benign", "Malignant"]  # Update if your folder order was different

st.title("🩺 Melanoma Skin Cancer Classification")
st.write("Upload a skin lesion image below to test your trained CNN model.")

uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Lesion Image", use_container_width=True)
    
    st.write("🔍 Analyzing image...")
    
    # Preprocess image
    image = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create batch dimension
    
    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.subheader("Prediction Results:")
    if predicted_class == "Malignant":
        st.error(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")
        st.warning("⚠️ This result indicates a potential malignancy. Please consult a dermatologist.")
    else:
        st.success(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")

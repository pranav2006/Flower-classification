import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

model = load_model('cnn_model.keras')

class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

st.title("🌸 Flower Classifier App")

uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    if st.button("🔍 Classify Image"):

        img = img.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(tf.nn.softmax(predictions)) * 100

        st.markdown(f"### Prediction: `{class_names[predicted_class]}`")
        st.markdown(f"### Confidence: `{confidence:.2f}%`")
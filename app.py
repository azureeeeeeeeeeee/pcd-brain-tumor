import streamlit as st
import numpy as np
import pickle
from PIL import Image
from utils import image_procesing_pipeline

model = None

with open("rfc.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Brain Tumor Classifier")
st.write("Upload gambar yang ingin diklasifikasikan.")

uploaded_file = st.file_uploader("Plih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    img = np.array(image)

    processed_img = image_procesing_pipeline(img, conv_type='sharpen', noise=False) # Preprocess gmbar agar mengikuti formast saat train


    input_data = processed_img.flatten().reshape(1, -1)
    pred = model.predict(input_data)[0]

    st.success(f"🧾 Predicted Tumor Type: **{pred}**")
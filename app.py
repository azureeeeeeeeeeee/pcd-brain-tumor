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

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

conv_type = st.selectbox(
    "Pilih metode konvolusi:",
    ("Original", "Edge Detection (Sobel X)", "Sharpen"),
    index=2
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # Map user selection to conv_type in utils.py
    conv_map = {
        "Original": "Original",
        "Edge Detection (Sobel X)": "Edge Detection (Sobel X)",
        "Sharpen": "Sharpen"
    }
    selected_conv = conv_map[conv_type]

    processed_img = image_procesing_pipeline(img, conv_type=selected_conv, noise=False)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar Asli", use_column_width=True)
    with col2:
        st.image(processed_img, caption="Setelah Preprocessing", use_column_width=True, clamp=True)

    input_data = processed_img.flatten().reshape(1, -1)
    pred = model.predict(input_data)[0]

    st.success(f"🧾 Predicted Tumor Type: **{pred}**")
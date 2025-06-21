import os
import requests
import streamlit as st
from PIL import Image
import numpy as np
# import cv2
import tempfile
from ultralytics import YOLO

st.cache_data.clear()
st.cache_resource.clear()

# === Page config must come first ===
st.set_page_config(page_title="Car Number Plate Detection", layout="centered")

# === Download model from Google Drive ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=1h_zsRpy2HSgXQMd-NoT8cYYmvYQtKHP2"
MODEL_PATH = "yolov8_car_plate.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return YOLO(MODEL_PATH)

# === Load model ===
model = load_model()

# === Streamlit UI ===
st.title("🚗 Car Number Plate Detection")
st.write("Upload a car image to detect number plates using YOLOv8.")

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Save temporarily for OpenCV use
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        img_cv = cv2.imread(tmp_file.name)

    st.info("Running detection...")

    # Run YOLO model
    results = model(img_cv)[0]

    # Show annotated image
    annotated_frame = results.plot()
    st.image(annotated_frame, caption="🔍 Detected Number Plate(s)", use_column_width=True)

    # Show cropped plates
    st.markdown("### 📌 Cropped Number Plate(s)")
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_plate = img_cv[y1:y2, x1:x2]

        if cropped_plate.size > 0:
            cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
            st.image(cropped_plate_rgb, caption=f"Plate #{i+1}", width=300)

        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()
        st.write(f"✅ Class ID: {cls_id} | Confidence: {conf:.2f}")

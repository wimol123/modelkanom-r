import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import detect

st.title("YUMMY")
st.write("Upload your Image...")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/Model2.pt')

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()))
    image = cv2.imdecode(file_bytes, 1)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the uploaded image
    st.image(imgRGB, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Detecting...")
    result = model(imgRGB, size=600)
    
    # Set a confidence threshold (adjust as needed)
    confidence_threshold = 0.5
    detect_class = result.xyxy[0][result.xyxy[0][:, 4] > confidence_threshold]
    
    st.code(detect_class[:, [6, 0, 2, 1, 3, 4]], caption=['name', 'xmin', 'xmax', 'ymin', 'ymax', 'confidence'])
    
    st.image(Image.fromarray(result.ims[0]), caption='Model Prediction(s)', use_column_width=True)

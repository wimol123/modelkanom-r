import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch
import cv2
import detect

st.title("YUMMY")

st.write("Upload your Image...")

# model = torch.hub.load('./yolov5', 'custom', path='./last.pt', source='local')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/Model2.pt')

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
  
  file_bytes = np.asarray(bytearray(uploaded_file.read()))
  image = cv2.imdecode(file_bytes, 1)

  imgRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
  # st.image(imgRGB)

  st.write("")
  st.write("Detecting...")
  result = model(imgRGB, size=600)
  
  detect_class = result.pandas().xyxy[0] 
  
  # แปลงค่า 'name' จาก "Darathong" เป็น "ดาราทอง"
  detect_class['name'] = detect_class['name'].map({'Darathong': 'ดาราทอง (Darathong)','SaneCharn': 'เสน่ห์จันทร์ (SaneCharn)','ChorMuang': 'ช่อม่วง (ChorMuang)'})
  
  st.code(detect_class['name'].drop_duplicates())
  
 # ใช้ st.image เพื่อแสดงภาพ "Darathong.jpg" ที่อัปโหลดมา
  st.image(Image.open("data/images/Darathong.jpg"), caption='Original Image', use_column_width=True)

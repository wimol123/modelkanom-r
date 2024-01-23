import argparse
import streamlit as st
import io
import os
from PIL import Image
import numpy as np
import torch, json , cv2 , detect


st.title("YUMMY")

st.write("Upload your Image...")

#model = torch.hub.load('./yolov5', 'custom', path='./last.pt', source='local')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/last.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/Model2.pt')

uploaded_file = st.file_uploader("Choose .jpg pic ...", type="jpg")
if uploaded_file is not None:
  
  file_bytes = np.asarray(bytearray(uploaded_file.read()))
  image = cv2.imdecode(file_bytes, 1)

  imgRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
  #st.image(imgRGB)

  st.write("")
  st.write("Detecting...")
  result = model(imgRGB, size=600)
  
  detect_class = result.pandas().xyxy[0] 
  
  #labels, cord_thres = detect_class[:, :].numpy(), detect_class[:, :].numpy()
  
  #     xmin       ymin    xmax        ymax          confidence  class    name
  #0  148.605362   0.0    1022.523743  818.618286    0.813045      2      turtle
  
 detect_class['name'] = detect_class['name'].map({'Darathong':'ดาราทอง (Darathong)','SaneCharn':'เสน่ห์จันทร์ (SaneCharn)','ChorMuang':'ช่อม่วง (ChorMuang)'})
  
  st.code(detect_class[['name']])
  
 # ใช้ st.image เพื่อแสดงภาพ "Darathong.jpg" ที่อัปโหลดมา
  st.image(Image.open("data/images/Darathong.jpg"), caption='Original Image', use_column_width=True)


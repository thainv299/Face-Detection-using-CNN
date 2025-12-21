import cv2
import os
from tensorflow.keras import models
import numpy as np
from matplotlib import pyplot as plt

# Load mô hình đã huấn luyện
model = models.load_model('YourModelPath')

# Tải bộ phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# Danh sách các tên lớp
list_name = []
dir = 'datasets/train-data'
for folder in os.listdir(dir):
    list_name.append(folder)

print(list_name)

# Đọc ảnh
cam = cv2.imread('your_image_path.jpg')

# Phát hiện khuôn mặt
faces = face_detector.detectMultiScale(cam, 1.3, 5)

for (x, y, w, h) in faces:
    # Cắt và thay đổi kích thước khuôn mặt
    frame = cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(frame[y:y + h, x:x + w], (128, 128))
    # Dự đoán lớp của khuôn mặt
    pred = np.argmax(model.predict(roi.reshape(1, 128, 128, 1)))
    print(pred)
    if np.max(model.predict(roi.reshape(1, 128, 128, 1))) > 0.3:  # Ngưỡng xác suất
        result = list_name[pred]
    else:
        result = 'Unknown'

    print(model.predict(roi.reshape(1, 128, 128, 1)))  # In ra giá trị dự đoán

    # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị kết quả
    cv2.rectangle(cam, (x, y), (x + w, y + h), (128, 255, 50), 1)
    cv2.putText(cam, result, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 25, 255), 2)

# Hiển thị ảnh với các khuôn mặt đã nhận diện
cv2.imshow("Detect", cam)
cv2.waitKey(0)

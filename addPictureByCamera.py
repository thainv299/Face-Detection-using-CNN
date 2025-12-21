import os
import cv2
import time

# Tải mô hình phát hiện khuôn mặt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Tạo thư mục lưu ảnh nếu chưa tồn tại
path = 'Picture'
if not os.path.exists(path):
    os.makedirs(path)

# Mở camera
cam = cv2.VideoCapture(0)
file_prefix = 'hinh'
i = 41  # Bộ đếm cho tên tệp
last_saved_time = time.time()  # Lưu thời gian lần chụp cuối
delay = 1  # Delay giữa các lần lưu ảnh (giây)
while True:
    ret, frame = cam.read()
    if not ret:
        print("Không thể lấy khung hình từ camera.")
        break

    # Chuyển đổi sang thang độ xám để phát hiện khuôn mặt
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)

    # Duyệt qua từng khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 255, 50), 3)

        # Chỉ lưu ảnh nếu đã qua thời gian delay
        current_time = time.time()
        if current_time - last_saved_time > delay:
            # Cắt và thay đổi kích thước khuôn mặt
            face_img = cv2.resize(frame_gray[y:y + h, x:x + w], (128, 128))
            # Tạo tên tệp và lưu ảnh
            filename = os.path.join(path, f"{file_prefix}_{i}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"Lưu {filename}")
            i += 1  # Tăng bộ đếm sau mỗi lần lưu
            last_saved_time = current_time  # Cập nhật thời gian lần chụp cuối

    # Hiển thị khung hình với các khuôn mặt được đánh dấu
    cv2.imshow("Face", frame)

    # Nhấn 'q' để thoát
    if (cv2.waitKey(1) == ord('q')):
        break

# Giải phóng camera và đóng cửa sổ
cam.release()
cv2.destroyAllWindows()

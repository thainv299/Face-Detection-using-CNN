import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras import models
from tkinter import  messagebox
model = models.load_model('model_FaceDetection.h5') #file model đã train
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')#công cụ phát hiện khuôn mặt
# Tạo thư mục
path = 'datasets/train-data' # đường dẫn tới dữ liệu train để lấy ra tên
listName =[]
def taoListName(path):
    listName = []
    for folder in os.listdir(path):
        listName.append(folder)
    return listName
listName = taoListName(path)
print("Mã hóa hoàn tất")
print(listName)
# Biến điều khiển camera
cap = None
camera_on = False

# Tạo giao diện Tkinter
window = tk.Tk()
window.title("Nhận diện khuôn mặt")
window.geometry("600x520")
window.update_idletasks()
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width // 2) - (window.winfo_width() // 2)
y = (screen_height // 2) - (window.winfo_height() // 2)
window.geometry(f"+{x}+{y}")
# Khung hình hiển thị video
lbl_video = Label(window)
lbl_video.place(x=62, y=20)

def start_camera():
    global cap, camera_on
    if not camera_on:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():  # Kiểm tra xem webcam có hoạt động không
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        camera_on = True
        update_frame()
def stop_camera():
    global cap, camera_on
    if camera_on:
        cap.release()
        cap = None
        camera_on = False
        lbl_video.config(image="")  # Xóa hình ảnh khi tắt camera

def update_frame():
    global cap, camera_on
    if camera_on and cap is not None:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc khung hình từ camera!")
            stop_camera()
            return
        frameS = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_detector.detectMultiScale(frameS,1.1,5)
        for (x,y,w,h) in face:
            roi = cv2.resize(frameS[y:y+h,x:x+w],(128,128))
            predict = model.predict(roi.reshape((-1,128,128,1)))
            print(predict)
            print(np.max(predict),listName[np.argmax(predict)])
            if np.max(predict) > 0.5 :
                result = listName[np.argmax(predict)]
            else :
                result = 'Unknow'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(128,250,50),2)
            cv2.putText(frame,result,(x,y),cv2.FONT_HERSHEY_PLAIN,2,(255,25,255),4)
        frame = cv2.resize(frame, (480, 360))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        lbl_video.after(10, update_frame)

def detect_from_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_detector.detectMultiScale(img_gray,1.1,5)
        for(x,y,w,h) in face :
            roi = cv2.resize(img_gray[y:y+h,x:x+w],(128,128))
            predict = model.predict(roi.reshape((-1,128,128,1)))
            print(predict)
            print(np.max(predict),listName[np.argmax(predict)])
            if np.max(predict) > 0.5 :
                result = listName[np.argmax(predict)]
            else :
                result = 'Unknow'
            cv2.rectangle(img,(x,y),(x+w,y+h),(128,250,50),2)
            cv2.putText(img,result,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,25,255),5)
        img = cv2.resize(img, (480, 360))
        cv2.imshow("DetectByImage", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
def detect_from_phoneCamera():
    cap1 = cv2.VideoCapture(1)
    while True:
        ret, frame = cap1.read()
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = face_detector.detectMultiScale(frame_gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(30,30))
        for (x,y,w,h) in face:
            roi = cv2.resize(frame_gray[y:y+h,x:x+w],(128,128))
            predict = model.predict(roi.reshape((-1,128,128,1)))
            print(predict)
            print(np.max(predict),listName[np.argmax(predict)])
            if np.max(predict) > 0.5:
                result = listName[np.argmax(predict)]
            else:
                result = 'Unknow'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(128,250,50),2)
            cv2.putText(frame,result,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,25,255),2)
        cv2.imshow("DetectFromIphoneCamera",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap1.release()
    cv2.destroyAllWindows()
def detect_from_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if file_path:
        video_cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face = face_detector.detectMultiScale(frame_gray,1.1,5)
            for(x,y,w,h) in face:
                roi = cv2.resize(frame_gray[y:y+h,x:x+w],(128,128))
                predict = model.predict(roi.reshape((-1,128,128,1)))
                print(predict)
                print(np.max(predict),listName[np.argmax(predict)])
                if (np.max(predict)) > 0.5:
                    result = listName[np.argmax(predict)]
                else :
                    result = 'Unknow'
                cv2.rectangle(frame,(x,y),(x+w,y+h),(128,250,50),2)
                cv2.putText(frame,result,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,25,255),5)
            frame = cv2.resize(frame, (480, 360))
            cv2.imshow("DetectByVideo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_cap.release()
        cv2.destroyAllWindows()

btn_start = tk.Button(window, text="Khởi động Camera", command=start_camera)
btn_start.place(x=50, y=400)

btn_stop = tk.Button(window, text="Ngắt Camera", command=stop_camera)
btn_stop.place(x=200, y=400)

btn_stop = tk.Button(window, text="Mở Camera điện thoại", command=detect_from_phoneCamera)
btn_stop.place(x=200, y=440)

btn_detect_image = tk.Button(window, text="Chọn ảnh để nhận diện", command=detect_from_image)
btn_detect_image.place(x=50, y=440)

btn_detect_video = tk.Button(window, text="Chọn video để nhận diện", command=detect_from_video)
btn_detect_video.place(x=50, y=480)

window.mainloop()

if cap:
    cap.release()
cv2.destroyAllWindows()

import os

import cv2
face_detector =cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
path = "YourPath"
makepath = 'DestinationPath'
if not os.path.exists(makepath):
    os.makedirs(makepath)
for img in os.listdir(path):
    faceImg = cv2.imread(f"{path}\{img}")
    faceImg = cv2.cvtColor(faceImg,cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(faceImg)
    for (x,y,w,h) in face:
        newFaceImg = cv2.resize(faceImg[y:y+h,x:x+w],(128,128))
        filename = os.path.join(makepath,img)
        cv2.imwrite(filename,newFaceImg)
print("Da cat anh thanh cong")

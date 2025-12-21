import cv2
import os
from tensorflow.keras import models
import numpy as np

models = models.load_model('YourModelPath')
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
list_name = []
dir = 'datasets/train-data'
for folder in os.listdir(dir) :
     list_name.append(folder)
print(list_name)
video = cv2.VideoCapture(0)
while True :
    ret,cam = video.read()
    face = face_detector.detectMultiScale(cam, 1.3, 5)
    frame = cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
    for (x,y,w,h) in face:
        roi = cv2.resize(frame[y:y+h, x:x+w], (128, 128))
        predict = models.predict(roi.reshape((-1, 128, 128, 1)))
        print(np.max(predict),list_name[np.argmax(predict)])
        if np.max(predict) > 0.5 :
            result = list_name[np.argmax(predict)]
        else :
             result = 'Unknow'
        print(predict)
        cv2.rectangle(cam,(x,y),(x+w,y+h),(128,255,50),1)
        cv2.putText(cam,result,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,25,255),2)
    cv2.imshow("detec",cam)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

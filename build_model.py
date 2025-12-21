import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import models
TRAIN_DATA = 'datasets/train-data' #Đường dẫn tới dữ liệu train
TEST_DATA = 'datasets/test-data'#Đường dẫn tới dữ liệu để testt
Xtrain = []
ytrain = []
# Xtrain = [(matranhinhanh1, ohe1), ... (matranhinhanhN,oheN)]

Xtest = []
ytest = []

dictionary = {'ALPacino': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Cuong': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              'ElonMusk': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'Hong' : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              'Khoa':    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              'Messi':   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'Ngoc' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              'Ronaldo': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Thai' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              'TomHiddle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              'testALPacino':  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'testCuong': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              'testElonMusk':  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'testHong' : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              'testKhoa':      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              'testMessi':     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'testNgoc' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              'testRonaldo':   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'testThai' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              'testTomHiddle': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               } # Đây là từ điển ánh xạ nhãn, sắp xếp 1 theo đúng thứ tự tên trong folder
def getData(dirData, lstData):
    for whatever in os.listdir(dirData):
        whatever_path = os.path.join(dirData,whatever)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path,filename)
            label = filename_path.split('\\')[1] # tach ten
            #img = cv2.imread(filename_path)
            img = np.array(Image.open(filename_path)) # chuyen hinh anh sang ma tran
            lst_filename_path.append((img,dictionary[label]))

        lstData.extend(lst_filename_path) # them tat ca phan tu o list lst_filename_path vao Xtrain
    return lstData

Xtrain = getData(TRAIN_DATA,Xtrain)
Xtest = getData(TEST_DATA,Xtest)
# Xtrain = [(x[0] / 255.0, x[1]) for x in Xtrain]
# Xtest = [(x[0] / 255.0, x[1]) for x in Xtest]
model_trainning1 = models.Sequential([
    layers.Conv2D(32, (3,3),input_shape=(128,128,1), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model_trainning1.summary()
model_trainning1.compile(optimizer= 'adam',
                         loss='categorical_crossentropy',
                         metrics= ['accuracy'])
model_trainning1.fit(
    np.array([x[0] for _, x in enumerate(Xtrain)]),
    np.array([y[1] for _, y in enumerate(Xtrain)]),
    epochs=30,
)
model_trainning1.save('model_FaceDetection.h5')

#test model
name_class = ['Al Pacino','Cuong','ElonMusk','Hong','Khoa','Messi', 'Ngoc', 'Ronadlo', 'Thai', 'TomHiddle']# danh sách tên sắp xếp theo đúng thứ tự
models = models.load_model('model_FaceDetection.h5')
np.random.shuffle(Xtrain)
acc = 0
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(Xtrain[i][0])
    if (np.argmax(models.predict(Xtrain[i][0].reshape((-1,128,128,1))))) == np.argmax(Xtrain[i][1]):
        acc += 1
    plt.title(name_class[np.argmax(models.predict(Xtrain[i][0].reshape((-1,128,128,1))))])
    plt.axis('off')
print(acc)
plt.show()

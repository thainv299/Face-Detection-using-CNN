import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

(Xtrain, ytrain), (Xtest, ytest) = cifar10.load_data()
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# [1],[6],[9] -> (one hot coding) : 0000000000,1000000000,0100000000,0010000000,0001000000

Xtrain , Xtest = Xtrain/255, Xtest/255
#ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)

# model_trainning1 = models.Sequential([
#     layers.Conv2D(32, (3,3),input_shape=(32,32,3), activation='relu'),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),
#
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.MaxPool2D((2, 2)),
#     layers.Dropout(0.2),
#
#     layers.Flatten(),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(10, activation='softmax'),
# ])
#model_trainning1.summary()
# model_trainning1.compile(optimizer= 'adam',
#                          loss='categorical_crossentropy',
#                          metrics= ['accuracy'])
#
# model_trainning1.fit(Xtrain, ytrain, epochs =10)

#model_trainning1.save("model-cifar10.h5")
modelx = models.load_model('model-cifar10.h5')

acc = 0
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(Xtest[500+i])
    if (np.argmax(modelx.predict(Xtest[500+i].reshape((-1,32,32,3)))) == ytest[500+i][0]) :
        acc = acc + 1
    plt.title(classes[np.argmax(modelx.predict(Xtest[500+i].reshape((-1,32,32,3))))])
    plt.axis('off')
plt.show()
print(acc)
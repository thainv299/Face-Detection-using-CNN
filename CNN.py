import matplotlib.pyplot as plt
import cv2
import numpy as np

np.random.seed(47)

img = cv2.imread("pic2\elon musk.jpg")
img = cv2.resize(img, (200, 200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
# print(img_gray)

class Conv2d:
    def __init__(self, img, numOfKernel=1,kernelSize=3,padding=0,stride=1):
        self.img = np.pad(img, ((padding, padding), (padding, padding)), 'constant')
        self.stride = stride
        self.kernel = np.random.randn(numOfKernel, kernelSize, kernelSize)
    # print(kernel)

        #output = (([z - k + 2 * padding]) /stride) +1
        self.result = np.zeros((int((self.img.shape[0] - self.kernel.shape[1])/self.stride) + 1,
                               int((self.img.shape[1] - self.kernel.shape[2])/self.stride) + 1,self.kernel.shape[0]))
    def getROI(self):
        for row in range(int((self.img.shape[0] - self.kernel.shape[1])/self.stride) + 1):
            for col in range(int((self.img.shape[1] - self.kernel.shape[2])/self.stride) + 1):
                roi = self.img[row*self.stride: row*self.stride + self.kernel.shape[1],
                      col*self.stride: col*self.stride + self.kernel.shape[2]]
                yield row,col,roi
    def operate(self):
        for layer in range(self.kernel.shape[0]):
            for row, col, roi in self.getROI():
                self.result[row, col, layer] = np.sum(roi * self.kernel[layer])
        return self.result
class Relu:
    def __init__(self,input):
        self.input = input
        self.result = np.zeros((self.input.shape[0],
                                self.input.shape[1],
                                self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0 if self.input[row, col, layer] < 0 else self.input[row, col, layer]

        return self.result
class LeakyRelu:
    def __init__(self,input):
        self.input = input
        self.result = np.zeros((self.input.shape[0],
                                self.input.shape[1],
                                self.input.shape[2]))

    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(self.input.shape[0]):
                for col in range(self.input.shape[1]):
                    self.result[row, col, layer] = 0.1*self.input[row,col,layer] if self.input[row, col, layer] < 0 \
                        else self.input[row, col, layer]

        return self.result
class MaxPooling:
    def __init__(self,input,poolingSize=2):
        self.input = input
        self.poolingSize = poolingSize
        self.result = np.zeros((int(self.input.shape[0]/self.poolingSize) ,
                                int(self.input.shape[1]/self.poolingSize) ,
                                self.input.shape[2]))
    def operate(self):
        for layer in range(self.input.shape[2]):
            for row in range(int(self.input.shape[0]/self.poolingSize)):
                for col in range(int(self.input.shape[1]/self.poolingSize)):
                    self.result[row, col, layer] = np.max(self.input[row * self.poolingSize : row * self.poolingSize + self.poolingSize ,
                                                              col * self.poolingSize : col * self.poolingSize + self.poolingSize ,
                                                              layer])
        return self.result

class Soft_max:
    def __init__(self, input, nodes):
        self.input = input
        self.nodes = nodes
        # y = w0 + w[i]*x
        self.flatten =self.input.flatten()
        self.weight = np.random.randn(self.flatten.shape[0])/self.flatten.shape[0]
        self.bias = np.random.randn(nodes)

    def operate(self):
        totals = np.dot(self.flatten(), self.weight) + self.bias
        exp = np.exp(totals)
        return exp/sum(exp)


img_gray_conv2d = Conv2d(img_gray).operate()
img_gray_conv2d_relu = Relu(img_gray_conv2d).operate()
img_gray_conv2d_relu_maxPooling = MaxPooling(img_gray_conv2d_relu).operate()
#img_gray_conv2d_Leakyrelu = LeakyRelu(img_gray_conv2d).operate()
#img_gray_conv2d_Leakyrelu_MaxPooling=MaxPooling(img_gray_conv2d_Leakyrelu,3).operate()


fig = plt.figure(figsize=(10,10))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.imshow(img_gray_conv2d_Leakyrelu_MaxPooling[:,:,i],cmap='gray')
#     plt.axis('off')
plt.imshow(img_gray_conv2d_relu_maxPooling,cmap='gray')
plt.axis('off')
#plt.savefig('img_gray_conv2d_Leakyrelu_MaxPooling.jpg')
plt.show()

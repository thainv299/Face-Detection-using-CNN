# Face Recognition using CNN and Haar Cascade
## Project Overview
This project implements a face recognition (classification) system using Convolutional Neural Networks (CNN).
The pipeline consists of two main stages:
Face Detection using OpenCV Haar Cascades
Face Classification using a CNN model trained on grayscale face images
The system detects faces from images or a camera stream, converts detected faces to grayscale (128×128), and then predicts the identity using a trained CNN model.
## Technologies Used
Python 3.10
OpenCV
TensorFlow / Keras
NumPy
Pillow (PIL)
Matplotlib
## Model Architecture (CNN)
The CNN model is built using Keras Sequential API with the following structure:
4 Convolutional Blocks:
    Conv2D + ReLU
    MaxPooling
    Dropout (to reduce overfitting)
Fully Connected Layers:
    Dense (1000 neurons)
    Dense (256 neurons)
    Output Dense layer with Softmax (number of classes)
```
Input shape:
(128, 128, 1)  # grayscale images
```

```
Loss function:
categorical_crossentropy
```

```
Optimizer:
Adam
```

## Dataset Structure
Your dataset should be organized as follows:
```
datasets/
│
├── train-data/
│   ├── ALPacino/
│   ├── Cuong/
│   ├── ElonMusk/
│   ├── ...
│
├── test-data/
│   ├── testALPacino/
│   ├── testCuong/
│   ├── testElonMusk/
│   ├── ...
```
## Important Notes
Folder names must exactly match the labels defined in the dictionary.
Images must be:
    Grayscale
    Size: 128×128

## Label Mapping (Dictionary)
Labels are encoded using One-Hot Encoding.
Example for 3 classes:
```
dictionary = {
    'A': [1, 0, 0],
    'B': [0, 1, 0],
    'C': [0, 0, 1],
    'testA': [1, 0, 0],
    'testB': [0, 1, 0],
    'testC': [0, 0, 1]
}
```
## Rules
Labels must be sorted alphabetically.
Folder names must match dictionary keys exactly.

## Image Collection & Preprocessing
### 1️⃣ Capture Face Images Using Camera
Use the script:
```
addPictureByCamera.py
```
This script:
Detects faces using Haar Cascade
Automatically crops faces
Converts them to grayscale
Resizes images to 128×128

### 2️⃣ Convert Existing Images
For images that are already captured, use:
```
ConvertImg.py
```
Parameters:

path: directory of original images
makepath: new directory to save processed images
The output folder will be created automatically inside your project directory.

## How to Run the Project
Step 1: Train the CNN Model
Run:
```
build_model.py
```
### Requirements
Install TensorFlow:
```
pip install tensorflow
```
### If you encounter the error "keras is not working", follow this tutorial:
👉 https://www.youtube.com/watch?v=mmfJyBJrGFU

You can also use the provided Keras library included in the project.
### Configuration Steps
Step 1: Update dataset paths
```
TRAIN_DATA = 'datasets/train-data'
TEST_DATA  = 'datasets/test-data'
```

Step 2: Update the dictionary according to your dataset
Ensure:

Correct order
Correct folder names

Output
After training:
The model is saved as:
```
model_FaceDetection.h5
```
### Model Evaluation (Optional)
The section #test model:
    Randomly samples training images
    Displays predictions
    Computes basic accuracy
### If you do not need this evaluation step, you can safely remove it.

Step 2: Run Face Detection & Recognition
Run:
```
FaceDetection.py
```
### Required Modifications
Step 1: Update model path
```
models.load_model('model_FaceDetection.h5')
```
Change this to the actual path of your trained model.
Step 2: Update Haar Cascade path
```
face_detector = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml'
)
```
- Make sure the path points correctly to the Haar Cascade XML file.
Step 3: Update image / video source path (if required)
Step 4: Run the script 

## Notes on Accuracy
- Model accuracy strongly depends on dataset size
- More images per person → higher accuracy
- Use consistent lighting conditions
- Always use grayscale images of size 128×128

## Future Improvements
- Replace Haar Cascade with MTCNN or RetinaFace
- Add data augmentation
- Use Transfer Learning (MobileNet, ResNet)
- Improve real-time FPS performance
- Add confidence threshold for predictions

## Author
Thai Nguyen
- GitHub: [thainv299](https://github.com/thainv299)

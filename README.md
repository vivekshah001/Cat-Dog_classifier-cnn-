# 🐶🐱 Cat-Dog_classifier-cnn
A Convolutional Neural Network (CNN) built to classify images of cats and dogs.
This project demonstrates end-to-end deep learning workflow: data preprocessing, model design, training, evaluation, and inference.

📌 Overview
This project trains a CNN model to distinguish between images of cats and dogs using supervised learning.

Problem Type: Binary Image Classification

Model Type: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras 


🧠 Model Architecture

Input Layer (150x150x3)

→ Conv2D + ReLU

→ MaxPooling

→ Conv2D + ReLU

→ MaxPooling

→ Conv2D + ReLU

→ MaxPooling

→ Flatten

→ Dense (Fully Connected)

→ Dropout

→ Output Layer (Sigmoid)

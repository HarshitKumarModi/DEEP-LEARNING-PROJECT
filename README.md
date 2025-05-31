# DEEP-LEARNING-PROJECT

COMPANY: CODETECH IT SOLUTIONS

NAME: HARSHIT KUMAR MODI

INTERN ID: CT04DN455

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION OF THE PROJECT:

🧠 Deep Learning Project – Image Classification Using CNN (CIFAR-10 Dataset)
This project was completed as part of my internship under CODTECH, with the objective of understanding and applying core deep learning techniques to solve a real-world image classification problem using Convolutional Neural Networks (CNNs). The project involves the classification of images from the widely used CIFAR-10 dataset using TensorFlow and its high-level Keras API.

📌 Project Overview
Deep learning is a powerful subset of machine learning inspired by the structure and function of the human brain's neural networks. Among the various applications of deep learning, image classification is one of the most popular and practical tasks. This project demonstrates how a Convolutional Neural Network (CNN) can be built and trained from scratch using TensorFlow to accurately classify color images into 10 categories.

The project uses the CIFAR-10 dataset, which contains 60,000 images of size 32x32 pixels, evenly distributed across 10 distinct classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is preloaded in TensorFlow, making it convenient to use for both experimentation and learning purposes.

🎯 Objectives
To gain hands-on experience in building deep learning models.

To understand the working of convolutional layers, pooling layers, and dense layers.

To preprocess image data for training a neural network.

To train and evaluate a CNN for multi-class image classification.

To visualize predictions and assess model performance.

To prepare a complete, functional project for academic or industrial presentation.

🛠️ Tools & Technologies
Programming Language: Python

Deep Learning Framework: TensorFlow / Keras

Visualization: Matplotlib

Development Environment: Google Colab / PyCharm

Dataset: CIFAR-10 (via tf.keras.datasets)

🧪 Methodology
1. Importing Required Libraries
We began by importing all necessary libraries including TensorFlow, NumPy for numerical operations, and Matplotlib for visualizing the results.

2. Loading and Preprocessing the Dataset
The CIFAR-10 dataset was loaded directly using TensorFlow’s built-in datasets module. Image pixel values were normalized to a range of 0 to 1 by dividing by 255.0. This normalization step is crucial for efficient and stable training of neural networks.

3. Data Visualization
To get a better sense of the data, we visualized 10 random training images with their class labels. This helped in understanding the image diversity and structure of the dataset.

4. Building the CNN Model
We constructed a sequential CNN model consisting of:

Three convolutional layers with ReLU activation functions

Two max-pooling layers for spatial dimensionality reduction

A flattening layer to convert the 2D outputs into 1D

Two dense layers, with the final layer having a softmax activation to output class probabilities

This model architecture is both lightweight and effective for image classification tasks on smaller datasets like CIFAR-10.

5. Compiling and Training the Model
The model was compiled using the Adam optimizer, with sparse categorical crossentropy as the loss function due to the use of integer class labels. The model was then trained using the fit() function for a few epochs with a batch size of 64.

6. Evaluating the Model
After training, the model was evaluated on the test dataset. The accuracy metric was used to quantify the performance. Typically, the model achieved 70–75% test accuracy after a few epochs of training, which is a reasonable baseline for a simple CNN on CIFAR-10.

7. Visualizing Predictions
To demonstrate the model’s performance, predictions were generated on test data. A set of sample images was displayed along with their predicted labels to visually validate the model's predictions.

📈 Results & Conclusion
The CNN model successfully learned to classify CIFAR-10 images with a decent level of accuracy. The project demonstrates how deep learning techniques can be effectively applied to solve classification problems. It also highlights the importance of model design, preprocessing, and evaluation in building machine learning pipelines.

This project helped in developing a strong foundational understanding of:

CNN architecture

Image data preprocessing

Model training and evaluation using TensorFlow

Visual representation of predictions

OUTPUT:

![Image](https://github.com/user-attachments/assets/740b8456-66aa-44b7-b164-7ea5448c9cf9)

![Image](https://github.com/user-attachments/assets/d309c3ff-0860-4c37-9324-df7c5298a69f)

![Image](https://github.com/user-attachments/assets/3eaa8e68-40d8-4a69-b231-f54d5ef84430)


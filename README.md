# Facial Emotion Detection Using Classical Computer Vision and Machine Learning

## Overview
This project implements a facial emotion detection system using classical computer vision techniques and machine learning classifiers. It processes grayscale facial images from the CK+ dataset, extracts visual features using SIFT and HOG, and applies supervised models to classify seven emotional states. The project also includes real-time detection and a bonus experiment using a webscraped dataset.

## Objectives
- Preprocess and standardize facial images
- Extract meaningful features using SIFT and HOG
- Train supervised models to classify emotions
- Evaluate performance using accuracy and classification reports
- Extend the model to real-time detection and webscraped data

## Dataset
- **Primary**: CK+ facial emotion dataset
- **Emotions**: Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise
- **Bonus**: Custom dataset scraped from Bing search results for each emotion

## Key Components

### 1. Image Preprocessing
- Resize all images to 128×128 pixels
- Optionally apply histogram equalization
- Organize by emotion categories for training

### 2. Feature Extraction
- **SIFT (Scale-Invariant Feature Transform)**: Keypoint detection and descriptor computation
- **HOG (Histogram of Oriented Gradients)**: Edge- and texture-based feature encoding
- Visual comparisons made to evaluate interpretability of both techniques

### 3. Classification
- Split dataset into training and testing sets
- Trained models:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors
- Metrics: Accuracy, confusion matrix, classification report

### 4. Real-Time Emotion Prediction
- Used OpenCV Haar cascades for face detection
- Applied trained SVM model to predict emotion from webcam input

### Webscraped Dataset
- Scraped images using `requests` and `BeautifulSoup` from Bing search
- Preprocessed and extracted HOG features
- Trained same models on the webscrapedd dataset

## Technologies Used
- Python
- Libraries:
  - OpenCV
  - scikit-learn
  - scikit-image
  - matplotlib
  - seaborn
  - BeautifulSoup, requests
- Tools: Jupyter Notebook

## Results
- **SVM achieved a very high accuracy of 100%* on the original CK+ dataset
- Real-time prediction works smoothly using OpenCV
- Experiment on the webscraped dataset demonstrates the importance of high-quality training data


# AuthentiCheck

## Video and Audio Deepfake Detection Project

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Preprocessing Method](#preprocessing-method)
4. [Tech Stack](#tech-stack)
5. [Setup Instructions](#setup-instructions)

## Overview

AuthentiCheck is a robust system designed for detecting deepfake videos and audio using advanced pretrained models. The project offers functionalities for video and audio analysis, including real-time livestream video processing. It leverages Streamlit for the web interface, PyTorch for video deepfake detection, and several other essential libraries.

## Features

### 1. Deepfake Video Detection

1. **Upload a Video File:**
   - Use the provided file upload widget to upload a video file (.mp4, .mov).

2. **Analyze Video for Deepfake Content:**
   - Click the "Check" button to analyze the uploaded video for deepfake content.
   - The application uses a pretrained VideoMAE model to perform the analysis.

3. **View Results:**
   - Displayed results include the predicted label ('REAL' or 'FAKE') and the confidence score indicating the model's prediction certainty.

![Deepfake Video](Images/i1.jpg)

### 2. Deepfake Audio Detection

1. **Upload an Audio File:**
   - Use the file upload widget to upload an audio file (.wav, .mp3).

2. **Analyze Audio for Deepfake Content:**
   - Click the "Check" button to analyze the uploaded audio file for deepfake content.
   - The application utilizes an audio-based detection model for this analysis.

3. **View Analysis Results:**
   - Results provide insights into the authenticity of the audio file, indicating if it contains suspicious content.

### 3. Livestream Video Analysis

1. **Access Webcam for Real-Time Analysis:**
   - Navigate to the "Livestream Video" section.
   - Grant permissions to allow the application to access your webcam.

2. **Perform Real-Time Deepfake Detection:**
   - The application starts streaming video from your webcam.
   - Real-time analysis is performed to detect deepfake content as the video streams.

![Livestream Analysis](Images/i2.jpg)

## Preprocessing Method
- **YOLOv8 Custom Fine-Tuned for Face Detection:**
    - We used YOLOv8, custom fine-tuned for face detection, to preprocess the video frames.
    - **OpenCV Built-In Object Tracking for Face Tracking:**
        - After detecting the face using YOLOv8, we utilized OpenCV's built-in object tracking for continuous face tracking across frames. This approach avoids the need for detecting the face in every frame, saving processing time and improving efficiency.

## Tech Stack

- **ViViT Transformer Model:** Used to extract intricate features from video data, ensuring high accuracy in deepfake detection.
- **Wav2vec Model:** Employed for analyzing audio features, adding an extra layer of verification and enhancing detection capabilities.
- **Late Fusion Technique:** Used for the fusion of video and audio features, allowing comprehensive and accurate classification.
- **Streamlit:** Utilized for deploying the solution, providing a user-friendly interface for both real-time and uploaded video analysis.
- **PyTorch:** Employed for implementing deep learning models

## Setup Instructions

### Prerequisites
- Python 3.6 or later
- pip package manager

### Notes:
- **requirements.txt:** Ensure your `requirements.txt` file lists all necessary libraries and versions required for the project. You can generate or update this file using `pip freeze > requirements.txt` after installing all dependencies.
  
- **Running the App:** The command `streamlit run app.py` starts the Streamlit application locally. Adjust `app.py` to match your actual Python script filename if different.
  
- **Web Interface:** Streamlit provides an intuitive web interface where users can interact with the application directly in their browser. Make sure to handle errors, exceptions, and user interactions gracefully within your Streamlit application logic.

![ViViT](Images/img_vivit.png)
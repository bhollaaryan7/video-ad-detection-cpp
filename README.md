# Video Advertisement Detection – C++ Deployment

## Overview

This project focuses on deploying video advertisement detection logic in C++, building on a machine learning pipeline developed in Python.

It demonstrates how trained models and feature-based detection can be implemented in a **low-level, efficient environment**, suitable for embedded systems.

---

## Key Components

### 1. Motion-Based Ad Detection
- Uses OpenCV optical flow (Farneback method)
- Detects high-motion segments as potential advertisements
- Applies thresholds and temporal smoothing

### 2. SVM Classification (Dlib)
- Features used:
  - Motion magnitude
  - Audio volume
- Trained using linear SVM
- Evaluated on a train-test split

---

## Tech Stack

- C++
- OpenCV
- Dlib

---

## Project Structure


video-ad-detection-cpp/
│
├── data/
│ └── sample/
│ ├── features.csv
│ └── labels.csv
│
├── src/
│ ├── motion_detection.cpp
│ ├── svm_classifier.cpp
│ └── test.cpp
│
├── results/
├── README.md
└── .gitignore


---

## How to Run

### 1. Basic Test (No Dependencies)

```bash
g++ -std=c++17 src/test.cpp -o test
./test
2. Dependencies Required

The following components require external libraries:

OpenCV → for motion detection
Dlib → for SVM classification
3. Example Compilation Commands
# SVM (requires Dlib)
g++ -std=c++17 src/svm_classifier.cpp -o svm -ldlib

# Motion detection (requires OpenCV)
g++ -std=c++17 src/motion_detection.cpp -o motion `pkg-config --cflags --libs opencv4`

Note: Ensure OpenCV and Dlib are installed and properly linked before compiling.

Key Learnings
Translating ML pipelines into C++ is non-trivial
Performance vs flexibility trade-offs
Challenges in embedded deployment environments
Importance of efficient feature-based models
Author

Aryan
MSc Machine Intelligence – University of Greenwich
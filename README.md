# Kannada Character Recognition Using Computer Vision and Machine Learning

## Overview
This project presents a robust system for recognizing handwritten Kannada characters using advanced computer vision and machine learning techniques. The project was developed as part of the credit requirements for our Foundations of Computer Vision and Machine Learning courses.

## Problem Statement
Handwritten character recognition plays a significant role in automating document digitization and enhancing human-computer interaction. However, challenges such as complex shapes in Kannada characters and varying handwriting styles make this task difficult. The objective of this project is to create a system capable of accurately classifying 50 distinct Kannada characters using various image processing and machine learning models.

## Project Pipeline
1. **Preprocessing the Images**:
   - **Inversion of Colors**: Inverts image colors to enhance contrast.
   - **Grayscale Conversion**: Reduces image complexity.
   - **Thresholding**: Creates a binary image for clear outlines.
   - **Contour Extraction and Cropping**: Locates character boundaries and isolates them from noise.
   - **Resizing and Centering**: Ensures uniform image dimensions (64x64 pixels).
   - **Final Binarization**: Produces a clean, black-on-white character image.

2. **Feature Extraction**:
   - **Crack Code Feature Vector**: Represents boundary directions of characters with frequency counts and normalized histograms. Divides the image into 8x8 zones, creating a final feature vector of size 512.
   - **Histogram of Oriented Gradients (HOG)**: Extracts gradient-based features with cell size (8x8), block size (16x16), and 4 orientation bins for comparative analysis.

3. **Machine Learning Models**:
   - **Support Vector Machine (SVM)**:
     - Effective in high-dimensional spaces.
     - Robust with non-linear data using an RBF kernel.
     - Achieved the highest accuracy in recognizing Kannada characters.
   - **Decision Tree**:
     - Easy to interpret but prone to overfitting.
     - Moderate performance with nuanced character differentiation.
   - **k-Nearest Neighbors (k-NN)**:
     - Simple and effective for smaller datasets but computationally intensive for larger ones.
     - Performance drops with high-dimensional feature vectors.

4. **Front-End Integration**:
   - An interactive GUI using **Tkinter** allows users to draw characters and receive predictions.
   - Features buttons for prediction and canvas reset.

## Model Comparison
| Model                  | Accuracy       | Strengths                        | Weaknesses                       |
|------------------------|----------------|----------------------------------|----------------------------------|
| **SVM + HOG**  | **High (81%)** | Effective with high-dimensional data | Computationally intensive        |
| **Decision Tree**      | Moderate       | Easy to interpret                | Prone to overfitting             |
| **k-NN**               | Low-Moderate   | Simple implementation            | Computationally expensive        |

### Why SVM Performs Best:
- **Margin Optimization**: Finds the best hyperplane to separate classes.
- **Robust Kernels**: RBF kernel adapts well to non-linear character features.
- **Generalization**: Strong performance without overfitting.

## GUI Visualization
The GUI was built using **Tkinter**, providing a user-friendly interface where users can:
- Draw Kannada characters on a canvas.
- Predict the character class with the click of a button.
- Reset the canvas for new input.

  ![Screenshot 2024-11-05 181248](https://github.com/user-attachments/assets/55d8e9d2-e0be-432a-b7b2-6043caf58e00)

## Future Work
- Extend the system to recognize a broader range of Kannada characters (657 classes).
- Integrate camera/video input for real-time character recognition.
- Further optimize models for improved speed and accuracy.

## Conclusion
This project showcases how combining advanced feature extraction techniques with robust machine learning models can create an effective solution for handwritten Kannada character recognition.

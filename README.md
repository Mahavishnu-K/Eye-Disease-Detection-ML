# Eye Disease Detection

## Overview
This project is a deep learning-based system for detecting eye diseases from retinal images. It uses a fine-tuned **EfficientNet-B4** model to classify images into one of the following categories:
- Diabetic Retinopathy
- Healthy
- Pterygium
- Retinal Detachment
- Retinitis Pigmentosa

The project includes:
- Training and evaluation scripts.
- A pre-trained model for inference.
- Grad-CAM visualization for model interpretability.
- Integration with machine learning models (Random Forest and SVM) for feature-based classification.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Inference](#inference)
6. [Grad-CAM Visualization](#grad-cam-visualization)
7. [Feature Extraction and ML Models](#feature-extraction-and-ml-models)


---

## Requirements
To run this project, you need the following dependencies:
- Python 3.8 or higher
- PyTorch
- Torchvision
- OpenCV
- scikit-learn
- NumPy
- Matplotlib
- tqdm

You can install the required packages using the following command:
```bash
pip install torch torchvision opencv-python scikit-learn numpy matplotlib tqdm

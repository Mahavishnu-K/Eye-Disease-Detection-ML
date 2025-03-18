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

## Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset](#dataset)

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

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Eye-Disease-Detection.git
   cd Eye-Disease-Detection
   ```

2. You can install the required packages using the following command:
  ```bash
  pip install torch torchvision opencv-python scikit-learn numpy matplotlib tqdm
  ```

---

## Dataset
The dataset used for this project is the **Retinal Disease Classification Dataset**, which is publicly available on Mendeley Data. You can download the dataset from the following link:

[Retinal Disease Classification Dataset](https://data.mendeley.com/datasets/s9bfhswzjb/1)

### Dataset Structure
The dataset contains retinal images categorized into the following classes:
- **Diabetic Retinopathy**
- **Healthy**
- **Pterygium**
- **Retinal Detachment**
- **Retinitis Pigmentosa**

Each class has its own folder containing the corresponding images. The dataset should be organized as follows:
train/
├── Diabetic Retinopathy/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Healthy/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Pterygium/
│ └── ...
├── Retinal Detachment/
│ └── ...
└── Retinitis Pigmentosa/
└── ...

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import models
import os

class EfficientNetB4Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4Model, self).__init__()
        self.base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)  
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EfficientNetB4Model(num_classes=5) 
model.load_state_dict(torch.load("eye_disease_detection.pth", map_location=device))  
model = model.to(device) 
model.eval() 


class_names = [
    "Diabetic Retinopathy",
    "Healthy",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
]

def preprocess_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found!")

    img_height, img_width = 224, 224 
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0) 
    return img.to(device)  

def predict_eye_disease(image_path):
    img_tensor = preprocess_image(image_path)

    with torch.no_grad():  
        predictions = model(img_tensor)
        probabilities = torch.softmax(predictions, dim=1)  
        confidence, predicted_class = torch.max(probabilities, dim=1)

    confidence = confidence.item() * 100
    predicted_class = predicted_class.item()

    print(f"Predicted Disease: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

image_path = r"D:\Eye-Disease-Detection-main\src\train\Diabetic Retinopathy\DR79.jpg"  
predict_eye_disease(image_path)
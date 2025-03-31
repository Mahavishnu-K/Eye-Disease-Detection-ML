import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "eye_disease_detection.pth"

class EfficientNetB4Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4Model, self).__init__()
        self.base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)  
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = EfficientNetB4Model(num_classes=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

class_names = [
    "Diabetic Retinopathy",
    "Healthy",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa"
]

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img

def grad_cam(model, img_tensor, target_layer):
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_backward_hook(backward_hook)
    
    output = model(img_tensor)
    model.zero_grad()
    
    class_idx = output.argmax(dim=1)
    one_hot = torch.zeros_like(output)
    one_hot[0][class_idx] = 1
    output.backward(gradient=one_hot)
    
    hook_forward.remove()
    hook_backward.remove()
    
    activations = activations[0].squeeze()
    gradients = gradients[0].squeeze()
    
    pooled_gradients = torch.mean(gradients, dim=[1, 2])
    
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    return heatmap

def predict_and_visualize(image_path):
    img_tensor, original_img = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100
    
    print(f"Predicted Disease: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    
    target_layer = model.base_model.features[-1]
    heatmap = grad_cam(model, img_tensor, target_layer)
    
    original_img = original_img.resize((224, 224))
    original_img = np.array(original_img)
    
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + original_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM: {class_names[predicted_class]} ({confidence:.1f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

image_path = r"C:\Users\mahav\Downloads\Eye Disease Image Dataset\Original Dataset\Diabetic Retinopathy\DR957.jpg"
predict_and_visualize(image_path)
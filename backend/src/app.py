import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import io
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

class EfficientNetB4Model(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetB4Model, self).__init__()
        self.base_model = models.efficientnet_b4(weights=None)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetB4Model(num_classes=5).to(device)
model.load_state_dict(torch.load("eye_disease_detection.pth", map_location=device))
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

def img_to_base64(img_array):
    img_pil = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_tensor, original_img = preprocess_image(filepath)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item() * 100
        
        target_layer = model.base_model.features[-1]
        heatmap = grad_cam(model, img_tensor, target_layer)
        
        original_img = original_img.resize((224, 224))
        original_img = np.array(original_img)
        
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * 0.4 + original_img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        os.remove(filepath)
        
        return jsonify({
            'prediction': class_names[predicted_class],
            'confidence': f"{confidence:.2f}%",
            'original_image': img_to_base64(original_img),
            'grad_cam_image': img_to_base64(superimposed_img)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
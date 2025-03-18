import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os
from PIL import Image

DATA_PATH = r"train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5
MODEL_PATH = "eye_disease_detection.pth"

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=DATA_PATH, transform=transform)
val_dataset = ImageFolder(root=DATA_PATH, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.dataset.classes
print("Class names:")
print(class_names)

class EfficientNetB4Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4Model, self).__init__()
        self.base_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)  
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetB4Model(num_classes=NUM_CLASSES).to(device)
print(f"Using device: {device}")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Pre-trained model loaded successfully!")
else:
    print(f"No pre-trained model found at {MODEL_PATH}. Please train the model first.")
    exit()

def grad_cam(model, img_tensor, target_layer):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    output[:, output.argmax(dim=1)].backward()

    activations = activations[0].detach()
    gradients = gradients[0].detach()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap.cpu().numpy()

def visualize_gradcam(img_path, model, target_layer):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    heatmap = grad_cam(model, img_tensor, target_layer)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.4 + img
    plt.imshow(superimposed_img / 255)
    plt.axis('off')
    plt.show()

def extract_features(model, dataloader):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model.base_model.features(inputs)
            outputs = torch.flatten(outputs, start_dim=1)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

target_layer = model.base_model.features[-1]
visualize_gradcam(r"D:\Eye-Disease-Detection-main\src\train\Retinitis Pigmentosa\Retinitis Pigmentosa7.jpg", model, target_layer)

X_train, y_train = extract_features(model, train_loader)
X_test, y_test = extract_features(model, val_loader)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print("Random Forest Results:")
print(classification_report(y_test, rf.predict(X_test), target_names=class_names))

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)
print("SVM Results:")
print(classification_report(y_test, svm.predict(X_test), target_names=class_names))
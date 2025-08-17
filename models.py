import torch
import torch.nn as nn
from torchvision import models

class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224 -> 112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112 -> 56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56 -> 28
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(model_name, num_classes, weights_path, device):
    model = None
    if model_name == "CustomCNN":
        model = CustomCNN(num_classes)
        model.load_state_dict(torch.load(weights_path, map_location=device))

    elif model_name in ["ResNet152", "DenseNet201", "EfficientNetB3"]:
        if model_name == "ResNet152":
            model = models.resnet152(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "DenseNet201":
            model = models.densenet201(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name == "EfficientNetB3":
            model = models.efficientnet_b3(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        
        model.load_state_dict(torch.load(weights_path, map_location=device))

    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    model.to(device)
    model.eval()
    return model
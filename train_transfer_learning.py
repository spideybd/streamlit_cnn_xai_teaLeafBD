import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import argparse

DATA_DIR = 'teaLeafBD'
NUM_CLASSES = 7
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root=DATA_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_model(model_name, num_classes):
    model = None
    if model_name == "ResNet152":
        model = models.resnet152(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "DenseNet201":
        model = models.densenet201(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "EfficientNetB3":
        model = models.efficientnet_b3(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model.to(DEVICE)

def train_model(model, model_name):
    save_path = f"weights/{model_name.lower()}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model improved and saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a transfer learning model.")
    parser.add_argument('--model', type=str, required=True, choices=["ResNet152", "DenseNet201", "EfficientNetB3"])
    args = parser.parse_args()
    print(f"--- Training {args.model} ---")
    model_to_train = get_model(args.model, NUM_CLASSES)
    train_model(model_to_train, args.model)
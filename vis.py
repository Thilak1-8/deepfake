# classic deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
import torchvision

# additional utilities
from tqdm import tqdm
import time
import copy

# for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

DATA_DIR = 'dataset'
# real images are in /Real subfolder and fake in /Fake subfolder and csv in dataset.csv
DATA_DIR_REAL = os.path.join(DATA_DIR, 'Real')
DATA_DIR_FAKE = os.path.join(DATA_DIR, 'Fake')
CSV_FILE = os.path.join(DATA_DIR, 'dataset.csv')


"""
THAT WAS KINDA USELESS, Cause we have dedicated REAL and Fake Folders in the dataset
"""

pd.read_csv(CSV_FILE).head()
#get count of real and fake images from the directories
num_real_images = len(os.listdir(DATA_DIR_REAL))
num_fake_images = len(os.listdir(DATA_DIR_FAKE))
print(f'Number of real images: {num_real_images}')
print(f'Number of fake images: {num_fake_images}')

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#visualize a few images from the dataset
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Load datasets - use the parent directory instead
# ImageFolder expects structure: DATA_DIR/Real/*, DATA_DIR/Fake/*
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

print(f"Classes found: {dataset.classes}")
print(f"Total images: {len(dataset)}")

# Split into training , validation and testing sets

train_size = int(0.2 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
print(f'Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f'Number of batches in train loader: {len(train_loader)}')
#plot a few real and fake images in a 3x3 grid (original, non-transformed images)
# Create a dataset without transformations for visualization
transform_viz = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_viz = datasets.ImageFolder(root=DATA_DIR, transform=transform_viz)

# Split using the same seed for consistency
train_size = int(0.8 * len(dataset_viz))
val_size = len(dataset_viz) - train_size
train_dataset_viz, val_dataset_viz = torch.utils.data.random_split(
    dataset_viz, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

# Create loaders for visualization
train_loader_viz = DataLoader(train_dataset_viz, batch_size=32, shuffle=False)
val_loader_viz = DataLoader(val_dataset_viz, batch_size=32, shuffle=False)

# Get training images
dataiter = iter(train_loader_viz)
images, labels = next(dataiter)

# Create a 3x3 grid of training images
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for idx in range(9):
    npimg = images[idx].numpy()
    axes[idx].imshow(np.transpose(npimg, (1, 2, 0)))
    axes[idx].set_title(f'Train: {dataset.classes[labels[idx]]}', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
if not os.path.exists('images'):
    os.makedirs('images')
plt.savefig('images/train_samples.png', dpi=150, bbox_inches='tight')
print('Training samples saved to images/train_samples.png')
plt.show()

# Get validation images
dataiter_val = iter(val_loader_viz)
images_val, labels_val = next(dataiter_val)

# Create a 3x3 grid of validation images
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for idx in range(9):
    npimg = images_val[idx].numpy()
    axes[idx].imshow(np.transpose(npimg, (1, 2, 0)))
    axes[idx].set_title(f'Val: {dataset.classes[labels_val[idx]]}', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('images/val_samples.png', dpi=150, bbox_inches='tight')
print('Validation samples saved to images/val_samples.png')
plt.show()
# lets define a custom resnet model for this task 

from torchinfo import summary

class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# Instantiate the model, define loss function and optimizer
model = CustomResNet(num_classes=2).to(device)



summary(model, input_size=(batch_size, 3, 224, 224))
hyperparameters = {
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'weight_decay': 1e-4,
    
}

#TRAIN MODEL, USE VALIDATION SET FOR EARLY STOPPING, SAVE BEST MODEL BASED ON VAL ACCURACY

optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'],
                        weight_decay=hyperparameters['weight_decay'])

criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop with early stopping
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience = 5
patience_counter = 0

train_losses = []
train_accs = []
val_losses = []
val_accs = []

print('Starting training...')
for epoch in range(hyperparameters['num_epochs']):
    print(f'\nEpoch {epoch+1}/{hyperparameters["num_epochs"]}')
    print('-' * 50)
    
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc.cpu().item())
    
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
    
    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset)
    val_losses.append(val_epoch_loss)
    val_accs.append(val_epoch_acc.cpu().item())
    
    print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
    
    # Save best model
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Best model saved with val acc: {best_val_acc:.4f}')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f'\nEarly stopping triggered after {epoch+1} epochs')
        break
    
    scheduler.step()

print(f'\nTraining complete. Best val accuracy: {best_val_acc:.4f}')

# Load best model
model.load_state_dict(best_model_wts)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Acc')
ax2.plot(val_accs, label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('images/training_history.png', dpi=150, bbox_inches='tight')
print('Training history saved to images/training_history.png')
plt.show()
#load best model, and evaluate results on test set

import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.classes,
                yticklabels=dataset.classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('images/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('Confusion matrix saved to images/confusion_matrix.png')
    
    plt.show()
    
evaluate_model(model, test_loader)
# %pip install PyWavelets
# Advanced Feature Extraction: FFT, DCT, Wavelet, and Color Space Conversion
import cv2
import pywt
from scipy.fftpack import dct
from sklearn.utils.class_weight import compute_class_weight
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class AdvancedTransform:
    """Custom transform that applies advanced feature extraction techniques"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        
    def __call__(self, img):
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Resize
        img_resized = cv2.resize(img_np, (self.img_size, self.img_size))
        
        # 1. Color Space Conversion to YCbCr
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YCrCb)
        else:
            img_ycbcr = img_resized
        
        # 2. FFT Magnitude (on Y channel)
        y_channel = img_ycbcr[:, :, 0] if len(img_ycbcr.shape) == 3 else img_ycbcr
        fft = np.fft.fft2(y_channel)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        magnitude_spectrum = np.log(magnitude_spectrum + 1)  # Log scale
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
        
        # 3. DCT Coefficients (on Y channel)
        dct_coeff = dct(dct(y_channel.T, norm='ortho').T, norm='ortho')
        dct_coeff = (dct_coeff - dct_coeff.min()) / (dct_coeff.max() - dct_coeff.min())
        
        # 4. Wavelet Transform (on Y channel)
        coeffs = pywt.dwt2(y_channel, 'haar')
        cA, (cH, cV, cD) = coeffs
        wavelet_features = np.abs(cH)  # Using horizontal details
        wavelet_features = cv2.resize(wavelet_features, (self.img_size, self.img_size))
        wavelet_features = (wavelet_features - wavelet_features.min()) / (wavelet_features.max() - wavelet_features.min())
        
        # Stack all features as channels
        # Original YCbCr (3 channels) + FFT (1) + DCT (1) + Wavelet (1) = 6 channels
        fft_resized = cv2.resize(magnitude_spectrum, (self.img_size, self.img_size))
        dct_resized = cv2.resize(dct_coeff, (self.img_size, self.img_size))
        
        combined = np.stack([
            img_ycbcr[:, :, 0] / 255.0,  # Y channel
            img_ycbcr[:, :, 1] / 255.0,  # Cb channel
            img_ycbcr[:, :, 2] / 255.0,  # Cr channel
            fft_resized,
            dct_resized,
            wavelet_features
        ], axis=0)
        
        return torch.tensor(combined, dtype=torch.float32)

# Create datasets with advanced transformations
print("Creating datasets with advanced feature extraction...")
advanced_transform = AdvancedTransform(img_size=224)
dataset_advanced = datasets.ImageFolder(root=DATA_DIR, transform=advanced_transform)

# Split dataset
train_size = int(0.7 * len(dataset_advanced))
val_size = int(0.15 * len(dataset_advanced))
test_size = len(dataset_advanced) - train_size - val_size

train_dataset_adv, val_dataset_adv, test_dataset_adv = torch.utils.data.random_split(
    dataset_advanced, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f'Advanced dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}')

# Handle class imbalance by computing class weights
all_labels = [label for _, label in dataset_advanced]
class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f'\nClass distribution:')
unique, counts = np.unique(all_labels, return_counts=True)
for cls, count in zip(unique, counts):
    print(f'  {dataset_advanced.classes[cls]}: {count} images')

print(f'\nComputed class weights: {class_weights}')
print('These weights will be used in the loss function to handle class imbalance.')

# Create DataLoaders
batch_size_adv = 32
train_loader_adv = DataLoader(train_dataset_adv, batch_size=batch_size_adv, shuffle=True, num_workers=0)
val_loader_adv = DataLoader(val_dataset_adv, batch_size=batch_size_adv, shuffle=False, num_workers=0)
test_loader_adv = DataLoader(test_dataset_adv, batch_size=batch_size_adv, shuffle=False, num_workers=0)

print(f'\nDataLoaders created with batch size: {batch_size_adv}')
print(f'Number of batches - Train: {len(train_loader_adv)}, Val: {len(val_loader_adv)}, Test: {len(test_loader_adv)}')
# Define Custom ResNet Model for 6-channel input
from torchinfo import summary

class AdvancedResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(AdvancedResNet, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer to accept 6 channels instead of 3
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            6,  # 6 input channels (YCbCr + FFT + DCT + Wavelet)
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Initialize new conv1 weights
        with torch.no_grad():
            # Copy weights from original 3 channels and duplicate for other 3
            self.resnet.conv1.weight[:, :3, :, :] = original_conv1.weight
            self.resnet.conv1.weight[:, 3:, :, :] = original_conv1.weight
        
        # Modify final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

# Instantiate the advanced model
print("Creating Advanced ResNet model...")
model_adv = AdvancedResNet(num_classes=2, pretrained=True).to(device)

print("\nModel Architecture Summary:")
print("=" * 80)
summary(model_adv, input_size=(batch_size_adv, 6, 224, 224), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3)

# Count parameters
total_params = sum(p.numel() for p in model_adv.parameters())
trainable_params = sum(p.numel() for p in model_adv.parameters() if p.requires_grad)

print("\n" + "=" * 80)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
print("=" * 80)
# Training setup with class weights
hyperparameters_adv = {
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'weight_decay': 1e-4,
}

optimizer_adv = optim.Adam(model_adv.parameters(), 
                           lr=hyperparameters_adv['learning_rate'],
                           weight_decay=hyperparameters_adv['weight_decay'])

# Use weighted CrossEntropyLoss to handle class imbalance
criterion_adv = nn.CrossEntropyLoss(weight=class_weights)

scheduler_adv = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adv, mode='max', 
                                                      factor=0.5, patience=3, verbose=True)

# Training loop
best_val_acc_adv = 0.0
best_model_wts_adv = copy.deepcopy(model_adv.state_dict())
patience = 7
patience_counter = 0

train_losses_adv = []
train_accs_adv = []
val_losses_adv = []
val_accs_adv = []

print('Starting advanced model training...\n')
start_time = time.time()

for epoch in range(hyperparameters_adv['num_epochs']):
    print(f'\nEpoch {epoch+1}/{hyperparameters_adv["num_epochs"]}')
    print('-' * 80)
    
    # Training phase
    model_adv.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in tqdm(train_loader_adv, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer_adv.zero_grad()
        
        outputs = model_adv(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion_adv(outputs, labels)
        
        loss.backward()
        optimizer_adv.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset_adv)
    epoch_acc = running_corrects.double() / len(train_dataset_adv)
    train_losses_adv.append(epoch_loss)
    train_accs_adv.append(epoch_acc.cpu().item())
    
    print(f'Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
    
    # Validation phase
    model_adv.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader_adv, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model_adv(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion_adv(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
    
    val_epoch_loss = val_running_loss / len(val_dataset_adv)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset_adv)
    val_losses_adv.append(val_epoch_loss)
    val_accs_adv.append(val_epoch_acc.cpu().item())
    
    print(f'Val Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}')
    
    # Learning rate scheduling
    scheduler_adv.step(val_epoch_acc)
    
    # Save best model
    if val_epoch_acc > best_val_acc_adv:
        best_val_acc_adv = val_epoch_acc
        best_model_wts_adv = copy.deepcopy(model_adv.state_dict())
        torch.save(model_adv.state_dict(), 'best_model_advanced.pth')
        print(f'✓ Best model saved! Val Acc: {best_val_acc_adv:.4f}')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f'\n⚠ Early stopping triggered after {epoch+1} epochs')
        break

elapsed_time = time.time() - start_time
print(f'\n{"="*80}')
print(f'Training Complete!')
print(f'Time Elapsed: {elapsed_time//60:.0f}m {elapsed_time%60:.0f}s')
print(f'Best Validation Accuracy: {best_val_acc_adv:.4f}')
print(f'{"="*80}')

# Load best model
model_adv.load_state_dict(best_model_wts_adv)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(train_losses_adv, label='Train Loss', linewidth=2, marker='o', markersize=4)
ax1.plot(val_losses_adv, label='Val Loss', linewidth=2, marker='s', markersize=4)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss (Advanced Model)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs_adv, label='Train Acc', linewidth=2, marker='o', markersize=4)
ax2.plot(val_accs_adv, label='Val Acc', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Training and Validation Accuracy (Advanced Model)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/training_history_advanced.png', dpi=150, bbox_inches='tight')
print('\n✓ Training history saved to images/training_history_advanced.png')
plt.show()
# Comprehensive evaluation on test set

import seaborn as sns

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             classification_report, roc_auc_score)

def comprehensive_evaluation(model, test_loader, dataset_classes):
    """
    Perform comprehensive evaluation including:
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC Curve and AUC
    - Classification Report
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Print metrics
    print("\n" + "="*80)
    print("TEST SET EVALUATION RESULTS")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("="*80)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=dataset_classes, 
                                digits=4))
    
    # Create visualizations
    fig = plt.figure(figsize=(18, 5))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset_classes,
                yticklabels=dataset_classes,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14})
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j+0.5, i+0.7, f'({cm_normalized[i, j]*100:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    # 2. ROC Curve
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Metrics Bar Chart
    ax3 = plt.subplot(1, 3, 3)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    metrics_values = [accuracy, precision, recall, f1, roc_auc]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylim([0, 1.1])
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
    print('\n✓ Evaluation plots saved to images/comprehensive_evaluation.png')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

# Evaluate the advanced model
results = comprehensive_evaluation(model_adv, test_loader_adv, dataset_advanced.classes)

"""
dataset.py - CIFAR-10 Dataset with Albumentations
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# CIFAR-10 Statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10Dataset(Dataset):
    """Custom CIFAR-10 Dataset with Albumentations support"""
    
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        return image, label


def get_train_transforms():
    """Get training augmentations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1, 
            rotate_limit=15,
            p=0.5
        ),
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=16,
            min_width=16,
            fill_value=CIFAR10_MEAN,
            mask_fill_value=None,
            p=0.5
        ),
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])


def get_test_transforms():
    """Get test transforms"""
    return A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])


def get_dataloaders(batch_size=128, num_workers=2):
    """Create train and test dataloaders"""
    
    # Download CIFAR-10
    train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    test_data = torchvision.datasets.CIFAR10('./data', train=False, download=True)
    
    # Create datasets
    train_dataset = CIFAR10Dataset(
        train_data.data,
        train_data.targets,
        get_train_transforms()
    )
    
    test_dataset = CIFAR10Dataset(
        test_data.data,
        test_data.targets,
        get_test_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
"""
train.py - Training script for CIFAR-10
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse
from torchsummary import summary

from model import CIFAR10Net
from dataset import get_dataloaders
from utils import plot_metrics, save_checkpoint
from config import CONFIG


# Training history
train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    pbar = tqdm(loader, desc='Training')
    
    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        processed += len(data)
        
        pbar.set_description(f'Loss={loss.item():.4f} Acc={100*correct/processed:.2f}%')
    
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(loader))


def test_epoch(model, loader, criterion, device):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    
    test_losses.append(test_loss)
    test_acc.append(accuracy)
    
    print(f'Test: Loss={test_loss:.4f}, Acc={accuracy:.2f}%\n')
    return accuracy


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = CIFAR10Net().to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.summary:
        summary(model, (3, 32, 32))
    
    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)
    
    # Training setup
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # Training loop
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_epoch(model, train_loader, optimizer, criterion, device)
        acc = test_epoch(model, test_loader, criterion, device)
        scheduler.step()
        
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, acc, epoch)
        
        if acc >= 85.0:
            print(f'Target accuracy reached! Best: {best_acc:.2f}%')
            break
    
    # Save plots
    plot_metrics(train_losses, test_losses, train_acc, test_acc)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    
    main(args)
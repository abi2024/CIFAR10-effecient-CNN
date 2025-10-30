# CIFAR-10 Classification with Dilated Convolutions

## 📊 Project Overview
Implementation of a CNN for CIFAR-10 classification using dilated convolutions instead of MaxPooling, achieving **85%+ accuracy** with **<200k parameters**.

## 🏆 Key Achievements
- ✅ **85.2% Test Accuracy** achieved
- ✅ **95,452 Parameters** (under 200k limit)
- ✅ **Receptive Field: 55+** (requirement: >44)
- ✅ **200 Bonus Points** - Used dilated convolutions instead of MaxPooling

## 🏗️ Architecture

### Model Structure: C1→C2→C3→C4→GAP→FC
```
Input (32x32x3)
    ↓
C1 Block (3 convs): 3→10→10→16 channels [RF: 7]
    ↓
C2 Block (2 DSCs): 16→24→32 channels [RF: 11]
    ↓
C3 Block (2 dilated + 1 regular): 32→40→48→56 channels [RF: 31]
    ↓
C4 Block (2 dilated + 1 strided): 56→64→72→80 channels [RF: 55+]
    ↓
GAP: 80×16×16 → 80×1×1
    ↓
FC: 80 → 10 classes
```

### Key Features
1. **No MaxPooling**: Uses strided convolution (stride=2) in C4
2. **Depthwise Separable Convolutions**: Used in C2 for parameter efficiency
3. **Dilated Convolutions**: Progressive dilation (2→4→8) for RF expansion
4. **Global Average Pooling**: Reduces parameters significantly
5. **Batch Normalization**: After every convolution
6. **Dropout**: Progressive (0.01→0.02) for regularization

## 📈 Results

### Training Performance
- **Best Test Accuracy**: 85.2%
- **Best Train Accuracy**: 87.1%
- **Total Parameters**: 95,452
- **Epochs to 85%**: 76

### Training Log Summary
```
Epoch 1:  Train Acc: 28.94%, Test Acc: 37.82%
Epoch 20: Train Acc: 72.45%, Test Acc: 73.21%
Epoch 40: Train Acc: 79.82%, Test Acc: 78.93%
Epoch 60: Train Acc: 83.76%, Test Acc: 82.44%
Epoch 76: Train Acc: 87.12%, Test Acc: 85.21% ⭐
```

## 🔧 Requirements
```bash
torch==2.0.0
torchvision==0.15.0
albumentations==1.3.1
numpy==1.24.3
matplotlib==3.7.1
tqdm==4.65.0
torchsummary==1.5.1
```

## 🚀 Usage

### Training
```bash
python train.py --epochs 100 --batch_size 128 --lr 0.1
```

### Testing a Saved Model
```bash
python train.py --test --model_path outputs/best_model.pth
```

## 📁 Project Structure
- `model.py`: CNN architecture with dilated convolutions
- `dataset.py`: CIFAR-10 dataset class with augmentations
- `train.py`: Training and evaluation loops
- `utils.py`: Helper functions for visualization and metrics
- `config.py`: Hyperparameters and settings

## 🎯 Assignment Requirements Met

| Requirement | Status | Implementation |
|------------|--------|---------------|
| CIFAR-10 Dataset | ✅ | Custom dataset class with albumentations |
| C1C2C3C4 Architecture | ✅ | 4 convolution blocks as specified |
| No MaxPooling | ✅ | Stride=2 in C4 block |
| RF > 44 | ✅ | Achieved RF = 55+ |
| Depthwise Separable | ✅ | Used in C2 block |
| Dilated Convolution | ✅ | Used in C3 and C4 blocks |
| GAP | ✅ | AdaptiveAvgPool2d before FC |
| Augmentations | ✅ | HorizontalFlip, ShiftScaleRotate, CoarseDropout |
| 85% Accuracy | ✅ | 85.2% achieved |
| <200k Parameters | ✅ | 95,452 parameters |
| Code Modularity | ✅ | Separate modules for each component |
| 200 Bonus Points | ✅ | Dilated convolutions instead of MaxPool |

## 📊 Model Summary
```
Total params: 95,452
Trainable params: 95,452
Non-trainable params: 0
Input size (MB): 0.01
Forward/backward pass size (MB): 12.45
Params size (MB): 0.36
Estimated Total Size (MB): 12.82
```

## 👨‍💻 Author
Abishek Satnur

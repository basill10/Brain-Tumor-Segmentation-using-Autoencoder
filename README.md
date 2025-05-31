# Brain Tumor Segmentation using Autoencoder

This project implements an autoencoder-based deep learning model for brain tumor segmentation from MRI scans. The model uses a U-Net style architecture to perform pixel-wise segmentation of brain tumors from medical images.

## Project Overview

Brain tumor segmentation is a critical task in medical image analysis that helps radiologists and medical professionals identify and analyze tumor regions in MRI scans. This implementation uses a symmetric autoencoder with skip connections to achieve accurate segmentation results.

### Key Features

- **U-Net Architecture**: Symmetric encoder-decoder with skip connections
- **Custom Dice Loss**: Addresses class imbalance common in medical segmentation
- **Comprehensive Evaluation**: Dice coefficient and IoU metrics
- **Data Visualization**: Training progress and prediction visualization
- **Model Checkpointing**: Saves best performing model

## Dataset

The project uses the Brain Tumor Segmentation dataset from Kaggle:

- 3,065 PNG images with dimensions 512×512
- Grayscale MRI scans with corresponding binary masks
- Binary masks indicating tumor regions (white = tumor, black = background)

## Model Architecture

### Autoencoder Design

- **Input**: 256×256 grayscale MRI images
- **Encoder**: 4 encoding blocks with max pooling
- **Bottleneck**: 512 feature channels at 16×16 resolution
- **Decoder**: 4 decoding blocks with transposed convolutions
- **Skip Connections**: Concatenate encoder features with decoder features
- **Output**: 256×256 binary segmentation mask

### Network Details

- **Encoder**: 1 → 32 → 64 → 128 → 256 → 512 channels
- **Decoder**: 512 → 256 → 128 → 64 → 32 → 1 channels

## Implementation Details

### Preprocessing

- Convert RGB images to grayscale
- Resize to 256×256 pixels
- Normalize pixel values to [0,1] range
- Convert masks to binary format

### Loss Function

Custom Dice Loss implementation to handle class imbalance:

Dice Loss = 1 - (2 × intersection + smooth) / (pred² + target² + smooth)

### Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 8-32 (configurable)
- **Epochs**: 20
- **Data Split**: 70% train, 20% validation, 10% test
- **Device**: CUDA if available, else CPU

## Results

The model achieves competitive performance on brain tumor segmentation.

### Evaluation Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **Intersection over Union (IoU)**: Measures segmentation accuracy

### Visualization Features

- Training/validation loss curves
- Real-time prediction visualization during training
- Test set evaluation with 10 random samples
- Side-by-side comparison: Original | Ground Truth | Prediction

## Usage

### Prerequisites

```bash
pip install torch torchvision opencv-python numpy matplotlib pillow scikit-learn torchsummary kaggle tensorflow
```


├── brain_tumor_segmentation.py    # Main implementation
├── model.pth                  # Trained model weights
├── README.md                     # This file
└── visualizations/               # Training plots and sample predictions

## References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Dice Loss literature](https://paperswithcode.com/method/dice-loss)
- [Brain Tumor Segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)
```

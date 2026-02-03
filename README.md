# DFEM-Net: Dynamic Adaptive Feature Extraction Network (Unofficial Implementation)

[cite_start]This repository contains a PyTorch implementation of **DFEM-Net**, as proposed in the paper *"DFEM-Net: A dynamic adaptive feature extraction network based deep learning model for pedestrian and vehicle detection"*.

**Note:** This is a research implementation focused on the model architecture components. Pre-trained weights are not included.

## 📖 Overview

[cite_start]DFEM-Net is designed to improve object detection performance in complex traffic environments (e.g., fog, rain, crowds)[cite: 23]. [cite_start]It modifies the standard YOLOv8 architecture by introducing three key mechanisms to handle irregular object shapes and multi-scale challenges[cite: 54]:

1.  [cite_start]**TuaNet (Backbone):** Replaces standard convolution blocks with **Deformable Convolutions (DCN)** and a custom **TuaAttention** mechanism to adapt to irregular object shapes[cite: 56, 58].
2.  [cite_start]**Scalseq (Neck):** A multi-scale fusion module that uses **3D Convolutions** to mix features from different network depths (P3, P4, P5) simultaneously[cite: 60, 269].
3.  [cite_start]**Zoomcat (Neck):** A triple-branch module that splits features into Large, Medium, and Small scales to better detect densely overlapping small objects[cite: 62, 270].

## 🛠️ Architecture Components

### 1. TuaNet (Dynamic Backbone)
* [cite_start]**Purpose:** To solve the issue where fixed square kernels fail to capture irregular vehicle/pedestrian shapes[cite: 166].
* [cite_start]**Mechanism:** Uses Deformable Convolutions (DCN) to "learn" offsets for kernel sampling points[cite: 169].
* [cite_start]**Attention:** Incorporates `TuaAttention` with GELU activation and residual connections [cite: 238-241].

### 2. Scalseq (3D Feature Fusion)
* [cite_start]**Purpose:** To fuse features across scales (Small, Medium, Large) more effectively than standard concatenation[cite: 268].
* [cite_start]**Mechanism:** Stacks P3, P4, and P5 feature maps into a 3D tensor and applies `Conv3d` to learn inter-scale correlations[cite: 288, 307].

### 3. Zoomcat (Detail Booster)
* [cite_start]**Purpose:** To prevent small objects from being lost in deep layers[cite: 335].
* [cite_start]**Mechanism:** Processes features in parallel branches (Upsampled, Standard, Downsampled) and concatenates them to preserve both detail and semantic context[cite: 327, 345].

## 🚀 Installation

This implementation relies on `torch`, `torchvision`, and `ultralytics` (for the YOLOv8 base).

```bash
# Install PyTorch (ensure it matches your CUDA version if using GPU)
pip install torch torchvision

# Install Ultralytics for YOLOv8 utilities
pip install ultralytics

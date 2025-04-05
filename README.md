# ViT-from-Scratch

This repository demonstrates how to build a Vision Transformer (ViT) model from the ground up using PyTorch. It includes a comprehensive implementation of a Siglip-style vision transformer, complete with image preprocessing, patch embedding creation, attention mechanisms, MLP blocks, and transformer encoder assembly. This repository also shows how to load and verify pretrained weights from a Hugging Face model.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Description](#detailed-description)

---

## Overview

This repository provides a step-by-step guide on building a Vision Transformer from scratch. It covers the following key aspects:

- **Image Preprocessing:** Loading and preparing images using PIL and torchvision.
- **Patch Embedding:** Creating patch embeddings with convolution layers and adding positional embeddings.
- **Attention Mechanism:** Implementing both single-head and multi-head attention, including weight transfer from a Hugging Face pretrained model.
- **MLP and Encoder Layers:** Building MLP blocks and stacking encoder layers to form the complete transformer.
- **Model Assembly:** Combining embeddings, encoder layers, and post-layer normalization into a full Vision Transformer model.
- **Visualization:** Plotting patch embeddings before and after training to visualize the learned representations.

---

## Features

- **From-Scratch Implementation:** Learn the internal workings of Vision Transformers by building each component manually.
- **Pretrained Weight Transfer:** Compare your custom model's outputs with a pretrained Hugging Face model to ensure consistency.
- **Visualization:** Easily visualize patch embeddings to understand how images are represented within the transformer.
- **Modular Code:** Components such as embeddings, attention, and MLP modules are implemented as modular classes for clarity and reusability.

---

## Installation

### Prerequisites

- **Python 3.8 or later**
- **PyTorch:** Install via [PyTorch official website](https://pytorch.org/get-started/locally/)
- **Transformers:** Install via pip
- **Pillow:** For image loading and manipulation
- **Torchvision:** For image preprocessing and transformations
- **Matplotlib:** For visualizing embeddings

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Clone the repository

```bash
git clone https://github.com/eizadhamdan/ViT-from-Scratch.git
cd ViT-from-Scratch
```

### Run the script

```bash
python siglip.py
```

---

## Detailed Description

### Image Preprocessing

The script uses **PIL** to load images and **torchvision transforms** to preprocess them (resizing, normalization, and conversion to tensors). The function `preprocess_image` handles these tasks and prepares the image for the transformer model.

### Patch Embedding

Patch embeddings are created using a **convolution layer** that segments the image into non-overlapping patches. **Positional embeddings** are added to these patch embeddings to retain spatial information.

### Attention Mechanisms

- **Single-Head Attention:** The `Head` class implements a single attention head.
- **Multi-Head Attention:** The `MultiHeadAttention` class aggregates multiple single-head attention outputs and applies a final linear projection.
- **SiglipAttention Module:** This module integrates weight projections (`query`, `key`, `value`, and `output`) and computes **scaled dot-product attention**.

### MLP Block and Encoder Layers

- **MLP Block:** The `SiglipMLP` class implements a **feed-forward network** with two linear layers and a **GELU** activation.
- **Encoder Layers:** The `SiglipEncoderLayer` class stacks a **self-attention module** and an **MLP block** with **layer normalization** and **residual connections**.
- **Transformer Encoder:** The `SiglipEncoder` class creates a full encoder by **stacking multiple encoder layers**.

### Model Assembly

The final Vision Transformer is assembled in `SiglipVisionTransformer` and wrapped by `SiglipVisionModel`. The model is designed to **match the structure of the Hugging Face model**, facilitating **weight transfer** and **output verification**.

### Visualization

**Matplotlib** is used to visualize **patch embeddings** before and after passing through the transformer, providing insight into how the model **learns image representations**.

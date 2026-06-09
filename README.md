# Distorted Visual Sequence Pattern Recognition

This repository contains my solution for a deep learning challenge focused on recognizing and reconstructing text sequences from heavily distorted grayscale images. Standard Optical Character Recognition (OCR) systems typically fail on this dataset due to the complex visual artifacts. 

This project successfully solves the challenge, achieving a Character Error Rate (CER) of 0.4%.

## Problem Overview

Many modern digital systems struggle to read text when it is unclear, noisy, or intentionally distorted (similar to Captchas). The images in this dataset contain hidden text sequences affected by:
* Background noise and visual artifacts
* Overlapping symbols and shape deformation
* Blur, occlusion, and random patches
* Irregular spacing and alignment

The main goal is to design a model that learns meaningful visual patterns and accurately predicts the correct ordered sequence of characters, even under these difficult conditions.

## Model Architecture: CRNN

To tackle both the spatial distortions and the sequential nature of the text, I built a Convolutional Recurrent Neural Network (CRNN):

1. **CNN Backbone:** This acts as the feature extractor. A Convolutional Neural Network scans the distorted grayscale images and extracts robust visual features, effectively filtering out background noise and isolating character structures.
2. **BiLSTM (Bidirectional Long Short-Term Memory):** The extracted visual features are passed sequentially into a BiLSTM layer. The BiLSTM captures the context of the sequence from both left-to-right and right-to-left. This allows the network to accurately infer missing, blurred, or overlapping characters based on the surrounding letters.
3. **CTC Loss (Connectionist Temporal Classification):** The network is optimized using CTC loss, which is the standard approach for sequence recognition tasks where the exact alignment between the input image pixels and output text characters is unknown.

## Data Preprocessing and Augmentation

To ensure the model is highly resilient to the heavy distortions in the dataset:
* I utilized Albumentations for advanced image preprocessing and augmentation.
* Images were normalized and resized to a consistent dimension (32x128) to allow for stable batch processing.
* The character vocabulary was dynamically generated from the training labels, mapping every unique character to an index alongside a blank token required for CTC loss computation.

## Evaluation and Results

The official competition metric is the Character Error Rate (CER). CER is based on the Levenshtein Distance, calculating the minimum number of insertions, deletions, and substitutions required to perfectly match the predicted text to the ground truth.

Instead of relying on standard accuracy (which is a poor metric for sequential data), I implemented a custom function to compute the exact CER at the end of every epoch. 

### Training Highlights:
* **OneCycle Learning Rate:** I used a dynamic learning rate scheduler to speed up convergence and avoid getting stuck in local minima.
* **Model Checkpointing:** The training loop strictly monitored the validation CER. The model weights were only saved when an absolute new minimum CER was achieved.

### Final Result:
The model converged exceptionally well. The training and validation losses decreased smoothly, and the model achieved an approximate 0.0040 CER (0.4% error rate) on the validation set. This demonstrates that the CRNN architecture successfully learned to filter out the heavy distortions and reconstruct the hidden text sequences with near-perfect accuracy.

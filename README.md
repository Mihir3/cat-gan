# cat-gan 
<p align='center'>
<img src="https://github.com/Mihir3/cat-gan/blob/main/gan_samples/gan-mnist-3500.png" width="32%">
<img src="https://github.com/Mihir3/cat-gan/blob/main/gan_samples/food-gan-101.png" width="32%">
<img src="https://github.com/Mihir3/cat-gan/blob/main/gan_samples/cat-gan-12250.png" width="32%">
</p>

This repository contains the codebase for generating cat images using Generative Adversarial Networks (GANs). It was done as part of an assignment for **CS 444** at **UIUC**, taught by **Prof. Svetlana Lazebnik**.

## Repository Structure

### `gan/`
- **`losses.py`** – Implements three different GAN loss functions:
  - Classical GAN Loss
  - Least-Squares Loss (LSGAN)
  - Wasserstein Loss (WGAN)
- **`models.py`** – Defines the architecture for the **Generator** and **Discriminator**, inspired by [DCGAN](https://arxiv.org/pdf/1511.06434.pdf).
- **`train.py`** – Contains the main training loop for training the GAN model.
- **`utils.py`** – Includes preprocessing utilities and helper functions.

### `gan_mnist/`
This directory contains scripts for training a GAN on the **MNIST dataset**, primarily used for debugging and experimentation.

### `gan_train/`
This folder contains training scripts:
- **`cat_gan/`** – Standard Cat-GAN (using Classical GAN Loss).
- **`least-squared-cat-gan/`** – Cat-GAN trained with **Least-Squares Loss (LSGAN)**.
- **`wasserstein-cat-gan/`** – Cat-GAN trained with **Wasserstein Loss (WGAN)**.
- **`food-gan/`** (Extra Credit) – A variant trained to generate **food images** instead of cats.

## GAN Weights
- **Wasserstein-cat-gan**: [Weights](https://drive.google.com/drive/folders/1NXY2vYEWfYv1GB6T0R3CevIYpVxduQ_3?usp=drive_link)  
- **Food-gan**: [Weights](https://drive.google.com/drive/folders/1emMr8EF24HO7q1MjSaqHAGsZrd7AjzrY?usp=sharing) 

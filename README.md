# ISIC Skin Lesion Classification

An advanced Deep Learning project leveraging state-of-the-art Convolutional Neural Networks (CNNs) to classify skin lesions and assist in the early detection of skin cancers like Melanoma.

## 📌 Project Overview
This repository contains a computer vision solution trained on the ISIC dataset. By extracting and fusing features from several modern pre-trained deep learning architectures, the model optimizes its ability to classify dermoscopic skin images into respective diagnosis categories.

## 🛠️ Tech Stack & Key Architectures
- **Framework:** TensorFlow / Keras
- **Image Processing:** OpenCV, PIL
- **Base Models utilized for Transfer Learning:**
  - EfficientNet (B0)
  - ResNet50
  - VGG16

## 🚀 Key Implementation Details
1. **Feature Fusion:** The model architecture utilizes Keras layers to blend feature representations from different model backbones using `concatenate` and `GlobalAveragePooling2D` for optimal classification stability.
2. **Regularization:** Employs `Dropout` layers to mitigate model overfitting on high-dimensional clinical image features.
3. **Data Management:** Fully compatible with Google Colab and Google Drive workflows for cloud-based training.

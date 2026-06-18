# Multi-Model Deep Feature Fusion for Skin Lesion and Melanoma Classification

An advanced computer vision and medical imaging system built in Google Colab utilizing TensorFlow and Keras to automate the multi-class categorization of dermatological diseases. The architecture optimizes classification boundaries by extracting and merging features from multiple pre-trained deep convolutional neural network (CNN) backbones.

## 📊 Dataset Profile

The model is trained and evaluated using the **ISIC (International Skin Imaging Collaboration) Archive dataset**. The project assumes a structured cloud-directory organization map integrated via Google Drive.

* **Root Path Configuration:** `/content/drive/MyDrive/isic_dataset/`
* **Sub-Directory Splits:** * `train/` — Categorized into class-specific folders used for feature learning and weight adjustment.
  * `test/` — Held-out target evaluation folders to compute generalizability metrics.
* **Target Diagnostic Classes Implemented:**
  1. Melanoma (MEL)
  2. Melanocytic Nevus / Moles (NV)
  3. Basal Cell Carcinoma (BCC)
  4. Actinic Keratosis (AK)
  5. Benign Keratosis (BKL - Solar Lentigo / Seborrheic / Lichen-Planus)
  6. Dermatofibroma (DF)
  7. Vascular Lesion (VASC)
  8. Squamous Cell Carcinoma (SCC)

---

## 🧠 Model Architecture & Methodology

Instead of relying on a single neural network architecture, this implementation employs an **Ensembled Feature Fusion Backbone Pipeline**.
1. **Feature Extraction:** Images are simultaneously processed across varied architecture styles—specifically **EfficientNet (B0, B1, B2, B3)** for depth/resolution scaling efficiency, **ResNet50** for residual shortcut representations, and **VGG16** for simple localized patterns.
2. **Concatenation Layer:** The structural feature maps generated across these individual networks are systematically joined together utilizing Keras functional APIs.
3. **Pooling & Classification:** A `GlobalAveragePooling2D` layer flattens the multidimensional unified feature space. A dense classification head coupled with `Dropout` regularization calculates the probability distributions across the target skin disease classes using a `Softmax` output layer.

---

## 🛠️ Technology Stack & Dependencies

* **Framework Engine:** TensorFlow 2.x / Keras Functional API
* **Image Processing Pipeline:** OpenCV (`cv2`), PIL (`Image`), `ImageDataGenerator`
* **Data Management & Arrays:** NumPy, Pandas
* **Metrics Visualization:** Matplotlib, Seaborn
* **Runtime Platform:** Google Colab Environment with GPU Accel enabled

---

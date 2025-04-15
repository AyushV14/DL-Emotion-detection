
# 🎭 Emotion Detection from Facial Expressions using CNN

This project implements a deep learning-based model to detect **emotions from facial expressions** using the **FER-2013** dataset and a custom-built **Convolutional Neural Network (CNN)**. The model achieves **72.4% test accuracy**, and its lightweight architecture makes it suitable for deployment on **mobile and embedded devices**.

---

## 📚 Abstract

Emotion recognition is a key component in human-computer interaction, healthcare, education, and intelligent systems. This project presents a CNN-based classifier trained on the FER-2013 dataset to identify seven facial emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The model is optimized for efficiency and accuracy, making it ideal for real-time applications even on resource-constrained devices.

---

## 📂 Dataset

We use the **Facial Expression Recognition (FER-2013)** dataset available on Kaggle:

- 📎 [Dataset Link](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

The dataset contains 48x48 pixel grayscale facial images with one of the following labels:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 🧠 Model Overview

- Input: 48x48 grayscale image
- Architecture: Custom CNN
- Activation: ReLU
- Pooling: MaxPooling
- Regularization: Dropout
- Output Layer: Softmax (7 classes)
- Final Accuracy:
  - Training: ~75%
  - Validation: ~71%
  - Test: **72.4%**

---

## ⚙️ Preprocessing

1. **Normalization**: Pixel values scaled to [0, 1]
2. **Reshape**: Converted to (48, 48, 1) shape
3. **One-Hot Encoding** of labels
4. **Data Augmentation**:
   - Random rotation
   - Zoom
   - Horizontal flip

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- Scikit-learn

---

## 🚀 How to Use

### 1. Clone the repository
```bash
git clone https://github.com/your-username/emotion-detection-cnn.git
cd emotion-detection-cnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition), extract it, and place it in the root folder.

### 4. Train the model
```python
python train.py
```

### 5. Test the model
```python
python evaluate.py
```

### 6. Predict an emotion
```python
python predict.py --image path_to_image.jpg
```

---

## 📈 Results

- Best performance seen on **Happy** and **Neutral**
- Confusion matrix shows occasional misclassification between **Fear** and **Surprise**
- Lightweight design enables real-time inference

---

## 💡 Future Improvements

- Use **Grad-CAM** to visualize model attention
- Integrate webcam for live emotion detection
- Explore **transfer learning** (e.g., VGG, MobileNet)
- Handle **class imbalance** more effectively
- Extend to **multimodal emotion detection** (voice, text)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ✍️ Authors

- **Ayush Vikharankar**
- **Rena Shoby**
- **Shreyas Suryawanshi**

> Department of Information Technology  
> K.J. Somaiya School of Engineering, Mumbai, India

---


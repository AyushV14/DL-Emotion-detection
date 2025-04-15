# Emotion Detection using CNN

This project focuses on **emotion detection** from facial expressions using a **Convolutional Neural Network (CNN)**. The model is trained to recognize various emotional states from grayscale facial images, achieving an accuracy of approximately **55%**.

## 🔍 Overview

Facial expression recognition plays a crucial role in human-computer interaction, mental health analysis, and smart surveillance systems. In this project, we implement a deep learning-based solution to classify facial expressions into different emotions using a CNN model.

## 📂 Dataset

We use the publicly available **Facial Expression Recognition** dataset from Kaggle:

- 📎 [Dataset Link](https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition)

The dataset contains grayscale images of human faces categorized into various emotion classes such as:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 🧠 Model Architecture

The model is built using a **Convolutional Neural Network (CNN)**. Key components include:

- Multiple convolutional layers with ReLU activation
- Max-pooling layers
- Dropout layers for regularization
- Fully connected dense layers
- Softmax output layer for multi-class classification

> Final Accuracy: **~55%**

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Pandas
- Scikit-learn
- OpenCV (optional for preprocessing or visualization)

## 📈 Results

Despite the modest accuracy of 55%, the model demonstrates the potential of deep learning in facial emotion classification. With more fine-tuning, advanced architectures, or data augmentation, the accuracy can be further improved.

## 🚀 Future Improvements

- Implement data augmentation techniques
- Explore more advanced CNN architectures (e.g., ResNet, VGG)
- Use transfer learning from pretrained models
- Fine-tune hyperparameters and optimize training
- Add real-time emotion detection using a webcam feed

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome! Feel free to open issues or pull requests.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---


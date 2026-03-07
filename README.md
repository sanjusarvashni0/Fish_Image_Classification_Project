# 🐟 Fish Species Image Classification

## 📌 Overview
This project builds a deep learning model to classify fish species from images using **Convolutional Neural Networks (CNN)** and **Transfer Learning**.  
The dataset contains approximately **10,000 fish images** belonging to multiple categories.

Multiple pretrained models were evaluated to determine the best-performing architecture. A **Streamlit web application** was also developed to allow users to upload fish images and receive predictions.

---

## 🎯 Objectives
- Build a **CNN model from scratch**
- Apply **Transfer Learning**
- Compare multiple pretrained models
- Evaluate model performance
- Deploy the model using **Streamlit**

---

## 🧹 Data Preprocessing
- Image resizing to **224×224**
- Pixel value **rescaling to [0,1]**
- **Data augmentation**
  - Rotation
  - Zoom
  - Horizontal flip

These techniques improve model robustness and reduce overfitting.

---

## 🤖 Models Used
- Custom CNN
- **VGG16**
- **ResNet50**
- **MobileNet**
- **InceptionV3**
- **EfficientNetB0**

These models were originally trained on the **ImageNet dataset** and fine-tuned for fish classification.

---

## 📊 Model Evaluation
Models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Training and validation accuracy/loss were also visualized.

---

## 🚀 Deployment
A **Streamlit web application** was created where users can:

1. Upload a fish image
2. The model processes the image
3. The predicted fish species and confidence score are displayed

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Streamlit  

---

## ▶️ How to Run the Project

Clone the repository

```
git clone https://github.com/yourusername/fish-image-classification.git
```

Install dependencies

```
pip install -r requirements.txt
```

Run the Streamlit application

```
streamlit run app.py
```

---

## 👩‍💻 Author
**Sanju Sarvashni**

# 🐶🐱 Dog–Cat Image Classifier

A TensorFlow-based image classifier that predicts whether an image contains a **dog** or a **cat**.  
The project includes preprocessing, prediction, and visualization using OpenCV and Matplotlib.

---

## ⭐ **Features**
- Loads a trained TensorFlow `.h5` model  
- Automatically resizes and normalizes images  
- Predicts **dog** or **cat** with probability output  
- Displays the image with prediction + accuracy  
- Uses OpenCV & Matplotlib for visualization  

---

## 📁 **Project Structure**
finalProject/
│── ccn-train.py # Model training script
│── ccn-test.py # Image prediction script
│── cat_dog_classifier.h5 # (not included in repo)
│── train/ # Training dataset (excluded)
│── test/ # Test dataset (excluded)
│── .gitignore


⚠️ The model (.h5) and dataset folders are **excluded** due to GitHub file size limits.

---

## 🧠 **Requirements**
Install dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy
```
## 📜 **License**
This project is for educational and academic use.

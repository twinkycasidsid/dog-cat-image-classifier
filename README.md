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
| File/Directory | Description | Included in Repo? |
| :--- | :--- | :---: |
| `finalProject/` | Root directory of the project. | **Yes** |
| **ccn-train.py** | Script for **training** the CNN model. | **Yes** |
| **ccn-test.py** | Script for making **predictions** on new images. | **Yes** |
| `cat_dog_classifier.h5` | The **trained model weights** (HDF5 format). | **No** |
| `train/` | Directory containing the **training dataset** (images). | **No** |
| `test/` | Directory containing the **test/validation dataset** (images). | **No** |
| `.gitignore` | Specifies intentionally untracked files to ignore. | **Yes** |

⚠️ The model (.h5) and dataset folders are **excluded** due to GitHub file size limits.

---

## 🧠 **Requirements**
Install dependencies:

```bash
pip install tensorflow opencv-python matplotlib numpy
```
## 📜 **License**
This project is for educational and academic use.

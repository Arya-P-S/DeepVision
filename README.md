# DeepVision - Diabetic Retinopathy Detection

DeepVision is an AI-powered healthcare web application that detects Diabetic Retinopathy (DR) from retinal fundus images using Deep Learning and Flask.

---

## Features

- Detects Diabetic Retinopathy stages
- Upload retina images
- Deep Learning prediction using TensorFlow/Keras
- Flask-based web application
- Modern responsive UI
- Displays prediction confidence
- Image preview before prediction

---

## DR Classes

| Label | Class |
|------|------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Technologies Used

- Python
- TensorFlow / Keras
- Flask
- HTML / CSS
- NumPy
- Pandas
- Pillow
- Matplotlib

---

## Project Structure

```bash
DeepVision/
│
├── app.py
├── train_model.py
├── prepare_dataset.py
├── model.h5
├── train.csv
├── test.csv
│
├── train_images/
├── test_images/
│
├── dataset/
│   └── train/
│       ├── 0
│       ├── 1
│       ├── 2
│       ├── 3
│       └── 4
│
├── templates/
│   └── index.html
│
├── static/
│
├── accuracy.png
├── loss.png
├── confusion_matrix.png
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <your-github-link>
cd DeepVision
```

### Install Dependencies

```bash
pip install tensorflow flask pandas pillow matplotlib seaborn scikit-learn
```

---

## Run Project

### Prepare Dataset

```bash
python prepare_dataset.py
```

### Train Model

```bash
python train_model.py
```

### Run Flask App

```bash
python app.py
```

---

## Open in Browser

```bash
http://127.0.0.1:5000/
```

---

## Model Outputs

- model.h5 → Trained model
- accuracy.png → Accuracy graph
- loss.png → Loss graph
- confusion_matrix.png → Confusion matrix

---

## Project Workflow

```text
Upload Image
      ↓
Image Preprocessing
      ↓
CNN Model Prediction
      ↓
Disease Classification
      ↓
Result Display
```

---

## Future Improvements

- Grad-CAM visualization
- Better pretrained models
- Online deployment
- Mobile application support
- Real-time hospital integration

---

## Author

Arya P S

---

## License

This project is for educational and research purposes.

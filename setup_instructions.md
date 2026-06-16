# Setup Instructions – Crop Disease Detection

## 1. Clone the Project

Clone the repository or download the project folder.

```
git clone <repository_link>
cd "Crop Disease Detection-clg"
```

If downloading manually, extract the project and navigate into the folder.

---

## 2. Install Python and Dependencies

Ensure **Python 3.9 – 3.11** is installed.

Check installation:
```
python --version
```

Create and activate a virtual environment:
```
python -m venv venv_crop
.\venv_crop\Scripts\activate
```

Install required libraries:
```
pip install -r requirements.txt
```

---

## 3. Download and Prepare the Datasets

Download a crop disease dataset such as **PlantVillage** from Kaggle or another dataset source.

Example source:
```
https://www.kaggle.com/datasets/emmarex/plantdisease
```

Prepare **two datasets** and one sample image folder:

### Training Dataset
Used for training the models.

### Evaluation Dataset
Used to evaluate model performance.

### Sample Web Images
A small collection of leaf images shown in the web app sidebar for quick testing.

Expected structure:
```
datasets/
    train_dataset/
        class_1/
        class_2/
        ...
    test_dataset/
        class_1/
        class_2/
        ...
    sample_web_images/
        image1.jpg
        image2.jpg
        ...
```

---

## 4. Update Dataset Paths in the Code

All paths are automatically calculated relative to the file location using `ROOT_DIR`. No manual path changes are needed if the folder structure above is followed correctly.

If you move any folder, update the relevant variable:

### In `src/train_model.py`
```
DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "train_dataset")
```

### In `src/eval_model.py`
```
TEST_DIR = os.path.join(ROOT_DIR, "datasets", "test_dataset")
```

### In `src/predict_xai.py`
```
dataset_dir = os.path.join(ROOT_DIR, "datasets", "train_dataset")
```

### In `src/backend/app.py`
```
SAMPLE_DIR = os.path.join(ROOT_DIR, "datasets", "sample_web_images")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "temp")
```

---

## 5. Train a Model

Navigate to the src folder:
```
cd src
```

Run:
```
python train_model.py
```

The training script contains multiple architectures:

- CNN
- MobileNetV2
- EfficientNetB0
- ResNet50

Only **one model section should be active at a time**.

To train a different model:
1. Comment the currently active model section.
2. Uncomment the desired model block.

Example model sections inside the script:
```
# with MobileNetV2
# with EfficientNetB0
# with ResNet50
# with CNN
```

After training, the model will be saved inside:
```
src/models/
```

Example saved model names:
```
best_crop_disease_model.keras
crop_disease_mobilenetv2_model.keras
crop_disease_efficientnet_model.keras
crop_disease_cnn_model.keras
crop_disease_resnet50_model
```

> **Note:** Model files are not included in the repository (excluded via `.gitignore`).
> After training, your models will be saved locally inside `src/models/`.
> Keep them safe — you will need to copy the path to use them in evaluation and the web app.

---

## 6. Evaluate a Model

Run:
```
python eval_model.py
```

Inside `eval_model.py`, update the model path to point to the model you just trained.

Copy the path of your saved model and update this line:
```
model = tf.keras.models.load_model("models/best_crop_disease_model.keras")
```

Replace with your model path. Examples:
```
model = tf.keras.models.load_model("models/crop_disease_mobilenetv2_model.keras")
model = tf.keras.models.load_model("models/crop_disease_cnn_model.keras")
model = tf.keras.models.load_model("models/crop_disease_efficientnet_model.keras")
model = tf.keras.models.load_model("models/crop_disease_resnet50_model")
```

The evaluation script outputs:

- Test Accuracy
- Classification Report
- Confusion Matrix
- Average Inference Time per Image

---

## 7. Run the Web Application

The web application connects the Flask backend, MobileNetV2 model, and Grad-CAM into a browser-accessible chat-style interface.

> **Before running**, make sure your trained model is saved at:
> ```
> src/models/best_crop_disease_model.keras
> ```
> If your model was saved with a different name, either rename it to `best_crop_disease_model.keras`
> or update this line in `src/predict_xai.py`:
> ```
> model = tf.keras.models.load_model(os.path.join(ROOT_DIR, "src", "models", "best_crop_disease_model.keras"))
> ```

Navigate to the backend folder:
```
cd src/backend
```

Run the Flask app:
```
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

### What the web app does:

- User uploads a leaf image via drag-and-drop or file browser
- Flask backend runs MobileNetV2 prediction and returns the disease class
- Grad-CAM heatmap is generated and overlaid on the leaf image
- Results displayed progressively in the chat-style UI
- Sample images from `datasets/sample_web_images/` shown in the sidebar for quick testing

### Flask API Routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend (index.html) |
| `/predict` | POST | Accepts uploaded image, returns prediction + heatmap path |
| `/samples` | GET | Returns list of sample images |
| `/samples/<file>` | GET | Serves a sample image |
| `/temp/<file>` | GET | Serves generated Grad-CAM heatmap |

---

## 8. Available Models in the Project

Trained models are stored in:
```
src/models/
```

> Model files are excluded from Git. After training, save all your models inside `src/models/`
> and copy their exact paths when switching between them in `eval_model.py` or `predict_xai.py`.

You can switch between models by modifying the model path in `eval_model.py`.

The web application always uses:
```
src/models/best_crop_disease_model.keras
```

This is the final MobileNetV2 model trained for 20 epochs with early stopping — **96% test accuracy, 0.030s inference time**.

---

## 9. Project Structure Reference

```
Crop Disease Detection-clg/
├── datasets/
│   ├── train_dataset/
│   ├── test_dataset/
│   └── sample_web_images/
├── src/
│   ├── backend/
│   │   ├── app.py
│   │   └── temp/
│   ├── frontend/
│   │   └── index.html
│   ├── models/
│   │   └── best_crop_disease_model.keras   ← not in Git, add manually after training
│   ├── predict_xai.py
│   ├── eval_model.py
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

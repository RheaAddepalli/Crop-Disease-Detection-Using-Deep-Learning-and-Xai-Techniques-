




# gto acccuracy 97 - made some changes in training - it have only per class so gonna add per crop . adding above this code 
import tensorflow as tf
import numpy as np
import os
import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ eval.py is in src/ → go one level up to reach project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ model is in src/models/
model = tf.keras.models.load_model(
    
    os.path.join(ROOT_DIR, "src", "models", "best_crop_disease_model.keras")
)

IMG_SIZE = 224

# ✅ test dataset is in datasets/test_dataset/
TEST_DIR = os.path.join(ROOT_DIR, "datasets", "test_dataset")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

# -------------------------
# Evaluate Accuracy
# -------------------------
loss, accuracy = model.evaluate(test_ds)
print("Test Accuracy:", accuracy)

# -------------------------
# Predictions
# -------------------------
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# -------------------------
# Classification Report
# -------------------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------------------------
# Per-Crop Accuracy
# -------------------------
print("\nPer-Crop Accuracy:\n")

crop_classes = {
    "Tomato": [i for i, c in enumerate(class_names) if c.startswith("Tomato")],
    "Potato": [i for i, c in enumerate(class_names) if c.startswith("Potato")],
    "Corn":   [i for i, c in enumerate(class_names) if c.startswith("Corn")]
}

for crop, indices in crop_classes.items():
    
    mask = np.isin(y_true, indices)

    crop_true = y_true[mask]
    crop_pred = y_pred[mask]

    correct = np.sum(crop_true == crop_pred)
    total = len(crop_true)

    acc = (correct / total) * 100

    print(f"{crop} Accuracy: {correct}/{total} = {acc:.2f}%")
# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# Latency Test
# -------------------------
sample_batch = next(iter(test_ds))[0]

start = time.time()
model.predict(sample_batch)
end = time.time()

latency = (end - start) / len(sample_batch)
print("Average Inference Time per Image (seconds):", latency)


















# 96
# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ✅ eval.py is in src/ → go one level up to reach project root
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # ✅ UPDATED MODEL PATH (matches training SAVE_PATH)
# model = tf.keras.models.load_model(
#     os.path.join(ROOT_DIR, "src", "models", "best_crop_disease_model_mobile_V1.keras")
# )

# IMG_SIZE = 224

# # ✅ TEST DATASET PATH
# TEST_DIR = os.path.join(ROOT_DIR, "datasets", "test_dataset")

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))

# # -------------------------
# # Per-Crop Accuracy
# # -------------------------
# print("\nPer-Crop Accuracy:\n")

# crop_classes = {
#     "Tomato": [i for i, c in enumerate(class_names) if c.startswith("Tomato")],
#     "Potato": [i for i, c in enumerate(class_names) if c.startswith("Potato")],
#     "Corn":   [i for i, c in enumerate(class_names) if c.startswith("Corn")]
# }

# for crop, indices in crop_classes.items():

#     mask = np.isin(y_true, indices)

#     crop_true = y_true[mask]
#     crop_pred = y_pred[mask]

#     correct = np.sum(crop_true == crop_pred)
#     total = len(crop_true)

#     acc = (correct / total) * 100

#     print(f"{crop} Accuracy: {correct}/{total} = {acc:.2f}%")

# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure(figsize=(12, 10))
# sns.heatmap(cm, annot=True, fmt="d",
#             xticklabels=class_names,
#             yticklabels=class_names)

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)












# # gto acccuracy 97 - made some changes in training this is  fro  percrop accuracy
# import tensorflow as tf
# import numpy as np
# import os
# import time
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ✅ eval.py is in src/ → go one level up to reach project root
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # ✅ model is in src/models/
# model = tf.keras.models.load_model(
    
#     os.path.join(ROOT_DIR, "src", "models", "best_crop_disease_model.keras")
# )

# IMG_SIZE = 224

# # ✅ test dataset is in datasets/test_dataset/
# TEST_DIR = os.path.join(ROOT_DIR, "datasets", "test_dataset")

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32,
#     shuffle=False
# )

# class_names = test_ds.class_names

# # -------------------------
# # Evaluate Accuracy
# # -------------------------
# loss, accuracy = model.evaluate(test_ds)
# print("Test Accuracy:", accuracy)

# # -------------------------
# # Predictions
# # -------------------------
# y_true = np.concatenate([y for x, y in test_ds], axis=0)
# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # -------------------------
# # Classification Report
# # -------------------------
# print("\nClassification Report:\n")
# print(classification_report(y_true, y_pred, target_names=class_names))
# # -------------------------
# # Crop-Level Classification Report
# # -------------------------
# print("\nCrop-Level Classification Report:\n")

# # Map class index → crop name
# index_to_crop = {}

# for i, name in enumerate(class_names):
#     if name.startswith("Tomato"):
#         index_to_crop[i] = "Tomato"
#     elif name.startswith("Potato"):
#         index_to_crop[i] = "Potato"
#     elif name.startswith("Corn"):
#         index_to_crop[i] = "Corn"
#     else:
#         index_to_crop[i] = "Other"

# # Convert y_true and y_pred to crop labels
# y_true_crop = np.array([index_to_crop[i] for i in y_true])
# y_pred_crop = np.array([index_to_crop[i] for i in y_pred])

# # Print classification report at crop level
# print(classification_report(y_true_crop, y_pred_crop))
# # -------------------------
# # Per-Crop Accuracy
# # -------------------------
# # -------------------------
# # TRUE Per-Crop Accuracy
# # -------------------------
# print("\nTrue Per-Crop Accuracy:\n")

# # Map class index → crop
# index_to_crop = {}

# for i, name in enumerate(class_names):
#     if name.startswith("Tomato"):
#         index_to_crop[i] = "Tomato"
#     elif name.startswith("Potato"):
#         index_to_crop[i] = "Potato"
#     elif name.startswith("Corn"):
#         index_to_crop[i] = "Corn"
#     else:
#         index_to_crop[i] = "Other"

# # Convert labels
# y_true_crop = np.array([index_to_crop[i] for i in y_true])
# y_pred_crop = np.array([index_to_crop[i] for i in y_pred])

# # Now calculate per-crop accuracy
# crops = ["Tomato", "Potato", "Corn"]

# for crop in crops:
#     mask = (y_true_crop == crop)

#     crop_true = y_true_crop[mask]
#     crop_pred = y_pred_crop[mask]

#     correct = np.sum(crop_true == crop_pred)
#     total = len(crop_true)

#     acc = (correct / total) * 100

#     print(f"{crop} Crop Accuracy: {correct}/{total} = {acc:.2f}%")
# # -------------------------
# # Confusion Matrix
# # -------------------------
# cm = confusion_matrix(y_true, y_pred)

# plt.figure()
# sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()

# # -------------------------
# # Latency Test
# # -------------------------
# sample_batch = next(iter(test_ds))[0]

# start = time.time()
# model.predict(sample_batch)
# end = time.time()

# latency = (end - start) / len(sample_batch)
# print("Average Inference Time per Image (seconds):", latency)










import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20

DATASET_DIR = r"C:\projects\Crop Disease Detection-clg\datasets\train_dataset"
SAVE_PATH = r"C:\projects\Crop Disease Detection-clg\src\models\best_crop_disease_model.keras"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"\nClasses ({len(class_names)}):", class_names)
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15)
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y)).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

base_model.trainable = False
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
print("\nPhase 1 — Training classifier head")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
print("\nPhase 2 — Fine tuning MobileNetV2")

# Unfreeze backbone
for layer in model.layers[:120]:
    layer.trainable = False

for layer in model.layers[120:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=[early_stop2, reduce_lr2, checkpoint]
)

print(f"\nBest model saved at: {SAVE_PATH}")
acc1 = history1.history["accuracy"]
vacc1 = history1.history["val_accuracy"]

acc2 = history2.history["accuracy"]
vacc2 = history2.history["val_accuracy"]

all_acc = acc1 + acc2
all_vacc = vacc1 + vacc2

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(all_acc,label="Train")
plt.plot(all_vacc,label="Validation")
plt.axvline(x=len(acc1)-1,color="gray",linestyle="--")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
loss1 = history1.history["loss"]
vloss1 = history1.history["val_loss"]

loss2 = history2.history["loss"]
vloss2 = history2.history["val_loss"]

plt.plot(loss1+loss2,label="Train")
plt.plot(vloss1+vloss2,label="Validation")
plt.axvline(x=len(loss1)-1,color="gray",linestyle="--")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()


































# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers.schedules import CosineDecay
# from matplotlib import pyplot as plt
# import os

# # -----------------------------
# # CONFIG
# # -----------------------------
# IMG_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS_HEAD = 10
# EPOCHS_FINETUNE = 30        # ✅ increased from 20

# DATASET_DIR = r"C:\projects\Crop Disease Detection-clg\datasets\train_dataset"
# SAVE_PATH = r"C:\projects\Crop Disease Detection-clg\src\models\best_crop_disease_model_mobile_V1.keras"

# # -----------------------------
# # LOAD DATASET
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE
# )

# class_names = train_ds.class_names
# num_classes = len(class_names)
# print(f"\nClasses ({num_classes}):", class_names)

# # -----------------------------
# # AUGMENTATION
# # -----------------------------
# augment = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.15),
#     layers.RandomZoom(0.15),
#     layers.RandomContrast(0.2),
#     layers.RandomBrightness(0.15),
#     layers.RandomTranslation(0.1, 0.1),  # ✅ NEW — helps with off-center leaves
# ], name="augmentation")

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.map(
#     lambda x, y: (augment(x, training=True), y)
# ).prefetch(AUTOTUNE)

# val_ds = val_ds.prefetch(AUTOTUNE)

# # -----------------------------
# # BUILD MODEL
# # -----------------------------
# inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# base_model = tf.keras.applications.MobileNetV2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )
# base_model.trainable = False

# # ✅ STRONGER CLASSIFIER HEAD
# x = layers.GlobalAveragePooling2D()(base_model.output)
# x = layers.BatchNormalization()(x)
# x = layers.Dense(512, activation="relu")(x)   # ✅ wider first layer
# x = layers.BatchNormalization()(x)             # ✅ extra BN helps convergence
# x = layers.Dropout(0.4)(x)
# x = layers.Dense(256, activation="relu")(x)   # ✅ second dense layer
# x = layers.Dropout(0.3)(x)
# outputs = layers.Dense(num_classes, activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # CALLBACKS (shared)
# # -----------------------------
# checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     SAVE_PATH,
#     monitor="val_accuracy",
#     save_best_only=True,
#     verbose=1
# )

# csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv")  # ✅ for paper table

# # =============================
# # PHASE 1 — Train Head Only
# # =============================
# print("\nPhase 1 — Training classifier head")

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# early_stop1 = tf.keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     patience=4,
#     restore_best_weights=True,
#     verbose=1
# )

# reduce_lr1 = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.3,
#     patience=2,
#     min_lr=1e-7,
#     verbose=1
# )

# history1 = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS_HEAD,
#     callbacks=[early_stop1, reduce_lr1, checkpoint, csv_logger]
# )

# # =============================
# # PHASE 2 — Fine Tuning
# # =============================
# print("\nPhase 2 — Fine tuning (deeper unfreeze)")

# # ✅ Unfreeze more layers — from 90 instead of 120
# for layer in model.layers:
#     layer.trainable = True

# for layer in base_model.layers[:90]:   # ✅ changed from 120 → 90
#     layer.trainable = False

# # ✅ Cosine decay — smoother than fixed LR, avoids late-epoch overshooting
# total_steps = EPOCHS_FINETUNE * len(train_ds)

# lr_schedule = CosineDecay(
#     initial_learning_rate=1e-5,
#     decay_steps=total_steps,
#     alpha=1e-7          # minimum LR at end of training
# )

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# early_stop2 = tf.keras.callbacks.EarlyStopping(
#     monitor="val_accuracy",
#     patience=8,                     # ✅ more patience for fine-tuning
#     restore_best_weights=True,
#     verbose=1
# )

# history2 = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS_FINETUNE,
#     callbacks=[early_stop2, checkpoint, csv_logger]
# )

# print(f"\nBest model saved to: {SAVE_PATH}")

# # -----------------------------
# # PLOT — both phases combined
# # -----------------------------
# acc  = history1.history["accuracy"]     + history2.history["accuracy"]
# vacc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
# loss = history1.history["loss"]         + history2.history["loss"]
# vloss= history1.history["val_loss"]     + history2.history["val_loss"]

# split = len(history1.history["accuracy"])  # where phase 1 ends

# fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# axes[0].plot(acc,  label="Train")
# axes[0].plot(vacc, label="Validation")
# axes[0].axvline(x=split - 1, color="gray", linestyle="--", label="Phase 1→2")
# axes[0].set_title("Accuracy"); axes[0].legend()

# axes[1].plot(loss,  label="Train")
# axes[1].plot(vloss, label="Validation")
# axes[1].axvline(x=split - 1, color="gray", linestyle="--", label="Phase 1→2")
# axes[1].set_title("Loss"); axes[1].legend()

# plt.tight_layout()
# plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
# print("✅ training_curves.png saved")
# plt.show()




























# current mobilenetv2 model with 97% accuracy - but making other changes for conference , did but this is the best 
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20

DATASET_DIR = r"C:\projects\Crop Disease Detection-clg\datasets\train_dataset"
SAVE_PATH = r"C:\projects\Crop Disease Detection-clg\src\models\best_crop_disease_model.keras"

# -----------------------------
# LOAD DATASET
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"\nClasses ({len(class_names)}):", class_names)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.15)
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y)).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# BUILD MODEL (MobileNetV2)
# -----------------------------
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=x
)

base_model.trainable = False

# -----------------------------
# CLASSIFIER HEAD
# -----------------------------
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------------
# COMPILE PHASE 1
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =============================
# PHASE 1 — Train Head
# =============================
print("\nPhase 1 — Training classifier head")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =============================
# PHASE 2 — Fine Tuning
# =============================
print("\nPhase 2 — Fine tuning MobileNetV2")

# Unfreeze backbone
for layer in model.layers[:120]:
    layer.trainable = False

for layer in model.layers[120:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr2 = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINETUNE,
    callbacks=[early_stop2, reduce_lr2, checkpoint]
)

print(f"\nBest model saved at: {SAVE_PATH}")

# -----------------------------
# PLOT TRAINING CURVES
# -----------------------------
acc1 = history1.history["accuracy"]
vacc1 = history1.history["val_accuracy"]

acc2 = history2.history["accuracy"]
vacc2 = history2.history["val_accuracy"]

all_acc = acc1 + acc2
all_vacc = vacc1 + vacc2

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(all_acc,label="Train")
plt.plot(all_vacc,label="Validation")
plt.axvline(x=len(acc1)-1,color="gray",linestyle="--")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
loss1 = history1.history["loss"]
vloss1 = history1.history["val_loss"]

loss2 = history2.history["loss"]
vloss2 = history2.history["val_loss"]

plt.plot(loss1+loss2,label="Train")
plt.plot(vloss1+vloss2,label="Validation")
plt.axvline(x=len(loss1)-1,color="gray",linestyle="--")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()






# current mobilenetv2 with 96 accuracy but got only 92
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\Crop Disease Detection-clg\datasets\train_dataset"

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR, validation_split=0.2, subset="training",
#     seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR, validation_split=0.2, subset="validation",
#     seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32)

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # ------------------------------

# # FIXED: BUILD FUNCTIONAL MODEL
# # ------------------------------
# inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")

# # preprocessing should NOT include separate InputLayers
# x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# base = tf.keras.applications.MobileNetV2(
#     include_top=False, weights="imagenet", input_tensor=x)

# x = layers.GlobalAveragePooling2D()(base.output)
# x = layers.Dropout(0.3)(x)
# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)
# model.compile(optimizer="adam",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])

# model.summary()

# model.fit(train_ds, validation_data=val_ds, epochs=20)

# # this model WILL work with Grad-CAM
# model.save("crop_disease_mobilenetv2_FIXED_FINAL.keras")
# print("✔ Saved fixed model")
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(['Train', 'Validation'])
# plt.show()



















# with MobileNetV2
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # MobileNetV2 MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for MobileNet
# x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# base_model = tf.keras.applications.MobileNetV2(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_mobilenetv2_model.keras")

# print("✔ MobileNetV2 model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("MobileNetV2 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()






#  # with EfficientNetB0
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # EfficientNet MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for EfficientNet
# x = tf.keras.applications.efficientnet.preprocess_input(inputs)

# base_model = tf.keras.applications.EfficientNetB0(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_efficientnet_model.keras")

# print("✔ EfficientNet model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("EfficientNetB0 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()





# # with resnet50
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # RESNET50 MODEL
# # -----------------------------

# inputs = tf.keras.Input(shape=(224,224,3))

# # Preprocess for ResNet
# x = tf.keras.applications.resnet50.preprocess_input(inputs)

# base_model = tf.keras.applications.ResNet50(
#     include_top=False,
#     weights="imagenet",
#     input_tensor=x
# )

# # Freeze base model
# base_model.trainable = False

# x = layers.GlobalAveragePooling2D()(base_model.output)

# x = layers.Dropout(0.3)(x)

# outputs = layers.Dense(len(class_names), activation="softmax")(x)

# model = tf.keras.Model(inputs, outputs)

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_resnet50_model")

# print("✔ ResNet50 model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("ResNet50 Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train","Validation"])

# plt.show()




# # with cnn
# from matplotlib import pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models

# IMG_SIZE = 224
# DATASET_DIR = r"C:\projects\cropdiseasedetection\dataset"

# # -----------------------------
# # Load Dataset
# # -----------------------------
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=42,
#     image_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=32
# )

# class_names = train_ds.class_names
# print("Classes:", class_names)

# # -----------------------------
# # CNN MODEL
# # -----------------------------
# model = models.Sequential([

#     layers.Rescaling(1./255, input_shape=(224,224,3)),

#     layers.Conv2D(32, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(64, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(128, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(256, (3,3), activation="relu"),
#     layers.MaxPooling2D(),

#     layers.GlobalAveragePooling2D(),

#     layers.Dropout(0.3),

#     layers.Dense(len(class_names), activation="softmax")
# ])

# # -----------------------------
# # Compile
# # -----------------------------
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # -----------------------------
# # Train
# # -----------------------------
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

# # -----------------------------
# # Save Model
# # -----------------------------
# model.save("crop_disease_cnn_model.keras")
# print("✔ CNN model saved")

# # -----------------------------
# # Accuracy Plot
# # -----------------------------
# plt.figure()
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])

# plt.title("CNN Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Validation"])

# plt.show()






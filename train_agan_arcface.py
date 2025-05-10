import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from agan_model import build_agan_model

# === CONFIG ===
IMG_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 25
DATASET_PATH = "/Users/saraholivia/Desktop/university/big_data_final/GhostFaceNets-main/faces_dataset"
MODEL_SAVE_PATH = "agan_small_arcface.h5"

# === DATA PREPROCESSING ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

num_classes = train_generator.num_classes

# === WRAP GENERATOR FOR MULTI-INPUT MODEL ===
def wrap_generator(generator):
    for batch_x, batch_y in generator:
        yield ({'image_input': batch_x, 'label_input': batch_y}, batch_y)

train_data = wrap_generator(train_generator)
val_data = wrap_generator(val_generator)

# === BUILD MODEL ===
model = build_agan_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, embedding_size=256)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# === TRAIN ===
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss')
]

model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    epochs=EPOCHS,
    callbacks=callbacks
) 

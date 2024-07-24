import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Directory where the dataset is stored
DATASET_DIR = 'D:/signlan/dataset'

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load and preprocess the data
def load_data(dataset_path):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        for img_file in os.listdir(os.path.join(dataset_path, label)):
            img_path = os.path.join(dataset_path, label, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            images.append(img)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load dataset
images, labels = load_data(DATASET_DIR)
images = images / 255.0  # Normalize pixel values

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the MobileNetV2 model, pre-trained on ImageNet and without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(lb.classes_), activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_test, y_test),
    epochs=10
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test accuracy: {accuracy*100:.2f}%')

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Save the trained model
model.save('sign_language_model.h5')




from sklearn.metrics import classification_report, confusion_matrix

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report
class_report = classification_report(y_true, y_pred_classes, target_names=lb.classes_)
print(class_report)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

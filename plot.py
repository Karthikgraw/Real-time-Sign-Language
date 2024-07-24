import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import datetime

# Directory where the dataset is stored
DATASET_DIR = 'D:/signlan/dataset'

# Gesture labels
gestures = ['I_Love_You', 'Victory', 'Okay', 'I_Dislike_It', 'Goodbye', 'Yes', 'LiveLong', 'Stop']

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Function to load a sample image from each gesture
def load_sample_images(dataset_dir, gestures):
    sample_images = []
    for gesture in gestures:
        gesture_dir = os.path.join(dataset_dir, gesture)
        sample_image_path = os.path.join(gesture_dir, np.random.choice(os.listdir(gesture_dir)))
        sample_image = cv2.imread(sample_image_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        sample_images.append(sample_image)
    return sample_images

# Function to load images and calculate the mean image
def load_and_mean_images(dataset_dir, gestures):
    mean_images = []
    for gesture in gestures:
        gesture_dir = os.path.join(dataset_dir, gesture)
        images = []
        for img_file in os.listdir(gesture_dir):
            img_path = os.path.join(gesture_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
        mean_image = np.mean(images, axis=0)
        mean_images.append(mean_image)
    return mean_images

# Function to count the number of images per gesture
def count_images_per_gesture(dataset_dir, gestures):
    gesture_counts = {gesture: len(os.listdir(os.path.join(dataset_dir, gesture))) for gesture in gestures}
    return gesture_counts

# Count the number of images per gesture
gesture_counts = count_images_per_gesture(DATASET_DIR, gestures)

# Plot the histogram of image distribution per gesture
plt.figure(figsize=(10, 6))
plt.bar(gesture_counts.keys(), gesture_counts.values(), color='skyblue')
plt.xlabel('Gesture')
plt.ylabel('Number of Images')
plt.title('Distribution of Images per Gesture')
plt.xticks(rotation=45)
plt.show()

# Plot sample images from each gesture
sample_images = load_sample_images(DATASET_DIR, gestures)
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
fig.suptitle('Sample Images from Each Gesture', fontsize=16)

for i, (gesture, image) in enumerate(zip(gestures, sample_images)):
    ax = axs[i // 4, i % 4]
    ax.imshow(image)
    ax.set_title(gesture)
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Plot the heatmap of average pixel intensities per gesture
mean_images = load_and_mean_images(DATASET_DIR, gestures)
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
fig.suptitle('Heatmap of Average Pixel Intensities per Gesture', fontsize=16)

for i, (gesture, mean_image) in enumerate(zip(gestures, mean_images)):
    ax = axs[i // 4, i % 4]
    sns.heatmap(mean_image, ax=ax, cmap='viridis')
    ax.set_title(gesture)
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Generate random data for the candlestick chart
np.random.seed(42)
num_points = len(gestures)
dates = pd.date_range(datetime.datetime(2024, 6, 1), periods=num_points)

# Create a dataframe for the candlestick chart
data = {'Date': dates,
        'Open': list(gesture_counts.values()),
        'High': list(gesture_counts.values()) + np.random.uniform(0, 5, num_points),
        'Low': list(gesture_counts.values()) - np.random.uniform(0, 5, num_points),
        'Close': list(gesture_counts.values()) + np.random.uniform(-2, 2, num_points)}

df = pd.DataFrame(data)
df['Date'] = df['Date'].map(mdates.date2num)

# Plot the candlestick chart
fig, ax = plt.subplots(figsize=(10, 6))
candlestick_ohlc(ax, df.values, width=0.6, colorup='g', colordown='r', alpha=0.8)

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator())

plt.title('Candlestick Chart for Gesture Dataset')
plt.xlabel('Date')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.show()



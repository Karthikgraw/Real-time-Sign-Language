import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

# Load the trained model
model = load_model('D:/signlan/sign_language_model.h5')

# Define the labels for the gestures
labels = ['I_Love_You', 'Victory', 'Okay', 'I_Dislike_It','Stop']
lb = LabelBinarizer()
lb.fit(labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Function to preprocess the frame for gesture recognition
def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

# Function to draw lines between finger landmarks
def draw_finger_lines(image, landmarks):
    # Define connections between finger landmarks
    finger_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]
    
    for connection in finger_connections:
        start_idx, end_idx = connection
        start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
        end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

# Start video capture
cap = cv2.VideoCapture(0)

print("Starting real-time hand tracking. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    # Draw hand landmarks and connections if any hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw custom finger lines
            draw_finger_lines(frame, hand_landmarks.landmark)

            # Preprocess the frame for gesture recognition
            img = preprocess_frame(frame)

            # Make predictions
            preds = model.predict(img)
            pred_label = lb.classes_[np.argmax(preds)]
            print(f"Predicted Label: {pred_label}")

            # Display the prediction on the frame
            cv2.putText(frame, pred_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Hand Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

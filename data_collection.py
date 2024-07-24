import cv2
import os
import time

# Dictionary to map gesture labels to directory names
gestures = {
    'i_love_you': 'I_Love_You',
    'victory': 'Victory',
    'okay': 'Okay',
    'i_dislike_it': 'I_Dislike_It',
    'stop': 'Stop'
}

# Number of images to capture per gesture
num_images = 100

# Function to capture images for a specific gesture
def capture_images(label, num_images):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not open webcam for {label}.")
        return
    
    count = 0

    # Create the directory if it doesn't exist
    gesture_dir = f'dataset/{gestures[label]}'
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f'Capturing images for {label}...')
    
    # Delay to give the user time to get ready
    print('Starting in 3 seconds...')
    time.sleep(3)

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the frame
        cv2.imshow('frame', frame)

        # Save the frame to the corresponding directory
        img_name = f'{gesture_dir}/{label}_{count}.jpg'
        cv2.imwrite(img_name, frame)
        count += 1

        # Print status to the console
        print(f"Captured image {count}/{num_images} for {label}.")

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Capture images for each gesture
for gesture in gestures.keys():
    capture_images(gesture, num_images)

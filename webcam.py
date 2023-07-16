import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import threading

# Threshold for hand confidence. Adjust this value as needed.
GESTURE_THRESHOLD = 0.1

hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

# Set the desired resolution and frame rate
WIDTH = 200  # Width of the frame (you can adjust as needed)
HEIGHT = 200  # Height of the frame (you can adjust as needed)
FPS = 15  # Frame rate per second (you can adjust as needed)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FPS, FPS)

# Function to simulate pressing the volume down key
def press_volume_down():
    keyboard.press('volume down')

# Function to simulate releasing the volume down key
def release_volume_down():
    keyboard.release('volume down')

# Function to simulate pressing the volume up key
def press_volume_up():
    keyboard.press('volume up')

# Function to simulate releasing the volume up key
def release_volume_up():
    keyboard.release('volume up')

# Function to detect and process gestures
def process_gesture():
    while cam.isOpened():
        success, img = cam.read()
        if not success:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.Hands(max_num_hands=2).process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(img, hand_landmarks, connections=hands.HAND_CONNECTIONS)

                # Check if fingers are touching (i.e., hand is closed)
                if hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[hands.HandLandmark.PINKY_MCP].y:
                    print("Decrease Volume")
                    press_volume_down()  # Simulate pressing the volume down key
                else:
                    print("Increase Volume")
                    press_volume_up()  # Simulate pressing the volume up key

        cv2.imshow("Koolac", img)
        if cv2.waitKey(100) & 0xFF == ord("q"):  # Limiting frame rate to ~10 FPS
            break

    # Release resources gracefully
    cam.release()
    cv2.destroyAllWindows()

# Create a separate thread for gesture processing
gesture_thread = threading.Thread(target=process_gesture)
gesture_thread.daemon = True  # This will allow the thread to stop when the main thread ends

# Start the gesture processing thread
gesture_thread.start()

# Main thread: Wait for user to press 'q' to exit
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Wait for the gesture processing thread to finish
gesture_thread.join()

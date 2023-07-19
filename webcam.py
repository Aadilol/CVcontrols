import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
import threading

# Threshold for hand confidence. Adjust this value as needed.
GESTURE_THRESHOLD = 0.1
FIST_DISTANCE_THRESHOLD = 0.1

hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils

# Set the desired resolution and frame rate
WIDTH = 320  # Width of the frame (you can adjust as needed)
HEIGHT = 240  # Height of the frame (you can adjust as needed)
FPS = 10  # Frame rate per second (you can adjust as needed)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FPS, FPS)

def press_volume_down():
    keyboard.press('volume down')

def press_volume_up():
    keyboard.press('volume up')

def press_space():
    keyboard.press('space')
    time.sleep(3)

def release_space():
    keyboard.release('space')

# detect and process gestures
def process_gesture():
    with hands.Hands(max_num_hands=1) as hand_detector:
        while cam.isOpened():
            success, img = cam.read()
            if not success:
                break

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hand_detector.process(img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(img, hand_landmarks, connections=hands.HAND_CONNECTIONS)

                    thumb_tip = np.array([hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP].x,
                                        hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP].y])
                    index_finger_tip = np.array([hand_landmarks.landmark[hands.HandLandmark.INDEX_FINGER_TIP].x,
                                                hand_landmarks.landmark[hands.HandLandmark.INDEX_FINGER_TIP].y])
                    middle_finger_tip = np.array([hand_landmarks.landmark[hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                                hand_landmarks.landmark[hands.HandLandmark.MIDDLE_FINGER_TIP].y])
                    ring_finger_tip = np.array([hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_TIP].x,
                                                hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_TIP].y])
                    pinky_tip = np.array([hand_landmarks.landmark[hands.HandLandmark.PINKY_TIP].x,
                                        hand_landmarks.landmark[hands.HandLandmark.PINKY_TIP].y])

                    # Calculate the distances between the thumb and fingers
                    thumb_index_distance = np.linalg.norm(thumb_tip - index_finger_tip)
                    thumb_middle_distance = np.linalg.norm(thumb_tip - middle_finger_tip)
                    thumb_ring_distance = np.linalg.norm(thumb_tip - ring_finger_tip)
                    thumb_pinky_distance = np.linalg.norm(thumb_tip - pinky_tip)
                    pinky_ring_distance = np.linalg.norm(pinky_tip - ring_finger_tip)

                    # Check if fingers are touching
                    if thumb_index_distance < GESTURE_THRESHOLD:
                        print("Decrease Volume")
                        press_volume_down()  #pressing the volume down key
                    elif thumb_index_distance > GESTURE_THRESHOLD:
                        print("Increase Volume")
                        press_volume_up()  #pressing the volume up key
                    # Check for open hand gesture
                    if (thumb_index_distance > FIST_DISTANCE_THRESHOLD and
                        thumb_middle_distance > FIST_DISTANCE_THRESHOLD and
                        thumb_ring_distance > FIST_DISTANCE_THRESHOLD and
                        thumb_pinky_distance > FIST_DISTANCE_THRESHOLD and
                        pinky_ring_distance > FIST_DISTANCE_THRESHOLD):
                        print("pause")
                        press_space()
                    else:
                        release_space()

            cv2.imshow("Koolac", img)
            if cv2.waitKey(100) & 0xFF == ord("q"):  # Limiting frame rate to ~10 FPS
                break

    # Release resources 
    cam.release()
    cv2.destroyAllWindows()

# Create a separate thread for gesture processing
gesture_thread = threading.Thread(target=process_gesture)
gesture_thread.daemon = True  # the thread to stops when the main thread ends


gesture_thread.start()

# Main thread
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

gesture_thread.join()

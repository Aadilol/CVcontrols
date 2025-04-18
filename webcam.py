import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import ctypes
import keyboard

# === CONFIG ===
COMMAND_COOLDOWN = 1
WIDTH, HEIGHT, FPS = 500, 500, 15
smoothing_factor = 0.25
move_threshold = 2
crop_ratio = 0.1  # Crop 10% from each edge of the camera frame

# === SETUP ===
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

mp_hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
last_command_time = {}
holding = False
smooth_x, smooth_y = 0, 0

# === ACTION MAPPING ===
gesture_action = {
    "Fist": "Mouse Click (1 hand) / Volume Down (2 hands)",
    "Pinky": "Volume Up",
    "Peace": "Next Track / Mouse Drag",
    "Open": "Right Click / Media Arm",
    "None": "",
    "IndexOnly": "Play/Pause / Mouse Move",
}

# === UTILITIES ===
def cooldown_passed(cmd):
    now = time.time()
    if cmd not in last_command_time or (now - last_command_time[cmd]) > COMMAND_COOLDOWN:
        last_command_time[cmd] = now
        return True
    return False

def volume_up(): keyboard.press_and_release('volume up')
def volume_down(): keyboard.press_and_release('volume down')
def pause_toggle(): keyboard.press_and_release('space')
def next_track():
    VK_MEDIA_NEXT_TRACK = 0xB0
    ctypes.windll.user32.keybd_event(VK_MEDIA_NEXT_TRACK, 0, 0, 0)
    ctypes.windll.user32.keybd_event(VK_MEDIA_NEXT_TRACK, 0, 2, 0)

def lerp(a, b, t): return a + (b - a) * t
def is_index_only(fup): return fup[1] == 1 and all(f == 0 for i, f in enumerate(fup) if i != 1)

# === GESTURE DETECTORS ===
def is_open(lm): return sum([
    lm[4][0] > lm[3][0],
    lm[8][1] < lm[6][1],
    lm[12][1] < lm[10][1],
    lm[16][1] < lm[14][1],
    lm[20][1] < lm[18][1],
]) == 5

def is_fist(lm):
    return all(lm[t][1] > lm[b][1] for t, b in [(8, 6), (12, 10), (16, 14), (20, 18)])

def is_peace(lm): return lm[8][1] < lm[6][1] and lm[12][1] < lm[10][1] and lm[16][1] > lm[14][1] and lm[20][1] > lm[18][1]

def fingers_up(lm):
    return [
        int(lm[4][0] > lm[3][0]),
        int(lm[8][1] < lm[6][1]),
        int(lm[12][1] < lm[10][1]),
        int(lm[16][1] < lm[14][1]),
        int(lm[20][1] < lm[18][1]),
    ]

def is_pinky_only(lm):
    return fingers_up(lm) == [0, 0, 0, 0, 1]

def snap_cursor(x, y):
    pyautogui.moveTo(x, y, duration=0)

# === MAIN LOOP ===
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        img = frame.copy()

        # Crop bounds for virtual zoom out
        left = int(w * crop_ratio)
        right = int(w * (1 - crop_ratio))
        top = int(h * crop_ratio)
        bottom = int(h * (1 - crop_ratio))

        hands_lm = []
        hand_states = []
        hand_positions = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = {i: np.array([pt.x, pt.y]) for i, pt in enumerate(hand_landmarks.landmark)}
                hands_lm.append(lm)
                hand_positions.append((int(lm[0][0] * w), int(lm[0][1] * h)))

                if is_fist(lm): gesture = "Fist"
                elif is_peace(lm): gesture = "Peace"
                elif is_pinky_only(lm): gesture = "Pinky"
                elif is_open(lm): gesture = "Open"
                else: gesture = "None"
                hand_states.append(gesture)

        # === DISPLAY: Gesture + Action
        for i, (pos, gesture) in enumerate(zip(hand_positions, hand_states)):
            action = gesture_action.get(gesture, "")
            fup = fingers_up(hands_lm[i])
            if gesture == "None" and is_index_only(fup):
                gesture = "IndexOnly"
                action = gesture_action["IndexOnly"]
            cv2.putText(img, f"{gesture} → {action}", (pos[0], pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # === ONE HAND (mouse control)
        if len(hands_lm) == 1:
            lm = hands_lm[0]
            fup = fingers_up(lm)
            gesture = hand_states[0]

            x = int(lm[8][0] * w)
            y = int(lm[8][1] * h)
            x = np.clip(x, left, right)
            y = np.clip(y, top, bottom)

            sx = np.interp(x, [left, right], [0, screen_width])
            sy = np.interp(y, [top, bottom], [0, screen_height])
            smooth_x = lerp(smooth_x, sx, smoothing_factor)
            smooth_y = lerp(smooth_y, sy, smoothing_factor)

            if is_index_only(fup):
                if holding:
                    pyautogui.mouseUp()
                    holding = False
                snap_cursor(smooth_x, smooth_y)

            elif fup == [0, 1, 1, 0, 0]:  # Peace = click & drag
                if not holding:
                    pyautogui.mouseDown()
                    holding = True
                snap_cursor(smooth_x, smooth_y)

            elif holding and not fup == [0, 1, 1, 0, 0]:  # Release on exit
                pyautogui.mouseUp()
                holding = False

            elif gesture == "Fist" and cooldown_passed("fist_click"):
                pyautogui.click()

        # === TWO HANDS (media control)
        elif len(hands_lm) == 2 and hand_states.count("Open") == 1:
            for i, (gesture, lm) in enumerate(zip(hand_states, hands_lm)):
                if gesture == "Open":
                    continue
                fup = fingers_up(lm)
                if is_index_only(fup):
                    gesture = "IndexOnly"
                    if cooldown_passed("pause"):
                        pause_toggle()
                elif gesture == "Fist" and cooldown_passed("vol_down"):
                    volume_down()
                elif gesture == "Pinky" and cooldown_passed("vol_up"):
                    volume_up()
                elif gesture == "Peace" and cooldown_passed("next"):
                    next_track()

        elif hand_states.count("Open") == 2:
            cv2.putText(img, "Both hands open — No action", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # === DISPLAY
        cv2.imshow("Gesture Mouse & Media Control", img)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

cap.release()
cv2.destroyAllWindows()

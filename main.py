import cv2
import mediapipe as mp
import math
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

hand_data = {"Left": {"prev_angle": None, "rotation_sum": 0, "direction": 0, "cooldown": 0, "show_until": 0, "is_rotating": False}, "Right": {"prev_angle": None, "rotation_sum": 0, "direction": 0, "cooldown": 0, "show_until": 0, "is_rotating": False}}


def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    wrist = hand_landmarks.landmark[0]

    folded_fingers = 0
    for tip_idx, pip_idx in zip(tips, pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        
        if get_distance(tip, wrist) < get_distance(pip, wrist):
            folded_fingers += 1

    return folded_fingers >= 3

def get_angle(hand_landmarks):
    thumb = hand_landmarks.landmark[2]  
    pinky = hand_landmarks.landmark[17] 

    dx = pinky.x - thumb.x
    dy = pinky.y - thumb.y

    return math.degrees(math.atan2(dy, dx))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    active_hand_labels = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            label = handedness.classification[0].label 
            active_hand_labels.append(label)
            
            fist = is_fist(hand_landmarks)
            angle = get_angle(hand_landmarks)
            data = hand_data[label]

            if fist:
                if data["prev_angle"] is not None:
                    diff = angle - data["prev_angle"]

                    if diff > 180: diff -= 360
                    elif diff < -180: diff += 360

                    if abs(diff) < 0.5:
                        diff = 0

                    diff *= 1.5

                    current_dir = 1 if diff > 0 else -1 if diff < 0 else 0

                    if data["direction"] != 0 and current_dir != data["direction"]:
                        data["rotation_sum"] = 0

                    if current_dir != 0:
                        data["direction"] = current_dir
                        data["rotation_sum"] += diff
                        data["prev_angle"] = angle

                    current_time = time.time()

                    if abs(data["rotation_sum"]) > 15 and not data["is_rotating"]:
                        data["is_rotating"] = True
                        data["show_until"] = current_time + 0.8
                        data["cooldown"] = current_time

                else:
                    data["prev_angle"] = angle

            else:
                data["prev_angle"] = None
                data["rotation_sum"] = 0
                data["direction"] = 0

            h, w, _ = frame.shape
            wrist_lm = hand_landmarks.landmark[0]
            cx, cy = int(wrist_lm.x * w), int(wrist_lm.y * h)
            
            status_text = f"{label} FIST" if fist else f"{label} HAND"
            cv2.putText(frame, status_text, (cx, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {int(angle)}", (cx, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Rot Sum: {int(data['rotation_sum'])}", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if data["is_rotating"]:
                current_time = time.time()
                if current_time < data["show_until"]:
                    cv2.putText(frame, f"{label} ROTATING!", (50, 100 if label == "Left" else 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    data["is_rotating"] = False
                    data["rotation_sum"] = 0
                    data["direction"] = 0

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for label in hand_data:
        if label not in active_hand_labels:
            hand_data[label]["prev_angle"] = None
            hand_data[label]["rotation_sum"] = 0
            hand_data[label]["direction"] = 0

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
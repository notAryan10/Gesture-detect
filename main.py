import cv2
import mediapipe as mp
import math
from functools import lru_cache

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

hand_data = {"Left": {"prev_angle": None, "rotation_sum": 0}, "Right": {"prev_angle": None, "rotation_sum": 0} }

@lru_cache(maxsize=128)
def get_distance(p1_coords, p2_coords):
    return math.sqrt(sum((a - b)**2 for a, b in zip(p1_coords, p2_coords)))

def is_fist(hand_landmarks):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    wrist = hand_landmarks.landmark[0]
    wrist_coords = (wrist.x, wrist.y, wrist.z)

    folded_fingers = 0
    for tip_idx, pip_idx in zip(tips, pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        
        tip_coords = (tip.x, tip.y, tip.z)
        pip_coords = (pip.x, pip.y, pip.z)

        if get_distance(tip_coords, wrist_coords) < get_distance(pip_coords, wrist_coords):
            folded_fingers += 1

    return folded_fingers >= 3

@lru_cache(maxsize=64)
def get_angle(wrist_coords, middle_coords):
    dx = middle_coords[0] - wrist_coords[0]
    dy = middle_coords[1] - wrist_coords[1]

    return math.degrees(math.atan2(dy, dx))


while True:
    get_distance.cache_clear()
    get_angle.cache_clear()
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
            
            wrist = hand_landmarks.landmark[0]
            middle = hand_landmarks.landmark[9]
            angle = get_angle((wrist.x, wrist.y), (middle.x, middle.y))
            data = hand_data[label]

            if fist:
                if data["prev_angle"] is not None:
                    diff = angle - data["prev_angle"]

                    if diff > 180: diff -= 360
                    elif diff < -180: diff += 360

                    data["rotation_sum"] += diff

                    if abs(data["rotation_sum"]) > 60:
                        cv2.putText(frame, f"{label} ROTATING!", (50, 100 if label == "Left" else 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        data["rotation_sum"] = 0 

                data["prev_angle"] = angle

            else:
                data["prev_angle"] = None
                data["rotation_sum"] = 0

            h, w, _ = frame.shape
            wrist_lm = hand_landmarks.landmark[0]
            cx, cy = int(wrist_lm.x * w), int(wrist_lm.y * h)
            
            status_text = f"{label} FIST" if fist else f"{label} HAND"
            cv2.putText(frame, status_text, (cx, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {int(angle)}", (cx, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Rot Sum: {int(data['rotation_sum'])}", (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    for label in hand_data:
        if label not in active_hand_labels:
            hand_data[label]["prev_angle"] = None
            hand_data[label]["rotation_sum"] = 0

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""Live rule-based gesture pipeline: webcam -> MediaPipe -> Spaces keystroke.

Detects swipes by wrist velocity over a short rolling window and fires the
matching macOS shortcut via osascript. No trained model needed.

Mapping (per proposal):
  swipe left  -> Ctrl+Right  next desktop      (hand pushes view right)
  swipe right -> Ctrl+Left   previous desktop  (hand pushes view left)
  swipe up    -> Ctrl+Up     Mission Control
  swipe down  -> Escape      dismiss Mission Control

Focus the video window and press q to quit.
"""

import collections
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np

BUFFER_FRAMES = 12
SWIPE_DX = 0.16
SWIPE_DY = 0.18
DOMINANCE = 1.2
COOLDOWN_S = 1.0
WRIST = 0

GESTURE_KEYS = {
    "swipe_left":  (124, "using control down"),
    "swipe_right": (123, "using control down"),
    "swipe_up":    (126, "using control down"),
    "swipe_down":  (53,  ""),
}


def fire(label):
    key_code, modifier = GESTURE_KEYS[label]
    print(f"[{time.strftime('%H:%M:%S')}] {label} -> key {key_code} {modifier}".rstrip())
    cmd = f'tell application "System Events" to key code {key_code} {modifier}'.strip()
    subprocess.run(["osascript", "-e", cmd], check=False)


def detect(buffer):
    if len(buffer) < BUFFER_FRAMES:
        return None
    arr = np.array(buffer)
    dx = arr[-1, 0] - arr[0, 0]
    dy = arr[-1, 1] - arr[0, 1]

    if abs(dy) > SWIPE_DY and abs(dy) > DOMINANCE * abs(dx):
        return "swipe_up" if dy < 0 else "swipe_down"
    if abs(dx) > SWIPE_DX and abs(dx) > DOMINANCE * abs(dy):
        return "swipe_right" if dx > 0 else "swipe_left"
    return None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    buffer = collections.deque(maxlen=BUFFER_FRAMES)
    last_fired = 0.0
    last_label = "ready"
    mc_open = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            wrist = hl.landmark[WRIST]
            buffer.append((wrist.x, wrist.y))
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        else:
            buffer.clear()

        now = time.time()
        if now - last_fired >= COOLDOWN_S:
            label = detect(buffer)
            if label == "swipe_up" and mc_open:
                label = None
            elif label == "swipe_down" and not mc_open:
                label = None

            if label is not None:
                fire(label)
                if label == "swipe_up":
                    mc_open = True
                elif label == "swipe_down":
                    mc_open = False
                last_fired = now
                last_label = label
                buffer.clear()

        cooldown_left = max(0.0, COOLDOWN_S - (now - last_fired))
        mc_status = "MC open" if mc_open else "MC closed"
        cv2.putText(frame, f"last: {last_label}   cooldown: {cooldown_left:.2f}s   {mc_status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "[r] reset MC state",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
        cv2.imshow("hands-off live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            mc_open = False

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()

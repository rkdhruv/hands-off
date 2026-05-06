"""Record landmark windows for gesture classifier training.

Captures webcam frames with OpenCV, runs MediaPipe Hands, and saves short
windows of hand landmarks (T frames x 21 landmarks x 3 coords) to
data/<label>/<timestamp>.npz.

Controls (focus the video window):
  1   select label: swipe_left
  2   select label: swipe_right
  3   select label: swipe_up
  4   select label: idle
  space  countdown then record one window for the selected label
  q   quit
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

WINDOW_FRAMES = 30
COUNTDOWN_SECONDS = 1.0
LABELS = {
    ord("1"): "swipe_left",
    ord("2"): "swipe_right",
    ord("3"): "swipe_up",
    ord("4"): "idle",
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WINDOW_NAME = "hands-off recorder"


def landmarks_to_array(hand_landmarks):
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )


def count_examples(label):
    d = DATA_DIR / label
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def draw_hud(frame, label, counts, status):
    lines = [
        f"label: {label}    count: {counts.get(label, 0)}",
        status,
        "[1]swipe_left [2]swipe_right [3]swipe_up [4]idle",
        "[space] record    [q] quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


def show(frame):
    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def countdown(cap, seconds, label):
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            return False
        frame = cv2.flip(frame, 1)
        remaining = end - time.time()
        cv2.putText(frame, f"recording {label} in {remaining:.1f}",
                    (10, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if show(frame) == ord("q"):
            return False
    return True


def record_window(cap, hands, mp_drawing, mp_hands, label):
    frames = []
    while len(frames) < WINDOW_FRAMES:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not result.multi_hand_landmarks:
            return None

        hl = result.multi_hand_landmarks[0]
        frames.append(landmarks_to_array(hl))

        mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"REC {label} {len(frames)}/{WINDOW_FRAMES}",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show(frame)

    return np.stack(frames)


def save_window(label, window):
    out_dir = DATA_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{int(time.time() * 1000)}.npz"
    np.savez_compressed(path, landmarks=window, label=label)
    return path


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("could not open webcam")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    label = "swipe_left"
    counts = {l: count_examples(l) for l in set(LABELS.values())}
    status = "ready"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        draw_hud(frame, label, counts, status)
        key = show(frame)

        if key == ord("q"):
            break
        if key in LABELS:
            label = LABELS[key]
            status = "ready"
        elif key == ord(" "):
            if not countdown(cap, COUNTDOWN_SECONDS, label):
                break
            window = record_window(cap, hands, mp_drawing, mp_hands, label)
            if window is None:
                status = "aborted: hand lost during window"
            else:
                path = save_window(label, window)
                counts[label] += 1
                status = f"saved {path.name}"

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()

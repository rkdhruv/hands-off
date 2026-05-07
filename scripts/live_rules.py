"""Live rule-based gesture pipeline: webcam -> MediaPipe -> Spaces keystroke.

Detects swipes by wrist velocity over a short rolling window and fires the
matching macOS shortcut via osascript. No trained model needed.

Mapping (per proposal):
  swipe left  -> Ctrl+Right  next desktop      (hand pushes view right)
  swipe right -> Ctrl+Left   previous desktop  (hand pushes view left)
  swipe up    -> Ctrl+Up     Mission Control
  swipe down  -> Escape      dismiss Mission Control
  fist        -> Spotify/Music play/pause via AppleScript
  double fist -> Spotify/Music skip to next track
  peace sign  -> Cmd+Shift+3  full-screen screenshot

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
COOLDOWN_S = 1.3
WRIST = 0

FINGER_TIP_MCP = [(8, 5), (12, 9), (16, 13), (20, 17)]
FIST_RATIO = 1.0
FIST_HOLD_FRAMES = 4
DOUBLE_FIST_WINDOW = 0.6

PEACE_HOLD_FRAMES = 4
PEACE_EXTENDED_RATIO = 1.4
PEACE_CURLED_RATIO = 1.0
STATIONARY_DISP = 0.05

GESTURE_KEYS = {
    "swipe_left":  (124, "using control down"),
    "swipe_right": (123, "using control down"),
    "swipe_up":    (126, "using control down"),
    "swipe_down":  (53,  ""),
    "peace":       (20,  "using {command down, shift down}"),
}

PLAYPAUSE_SCRIPT = '''
try
    if application "Spotify" is running then
        tell application "Spotify" to playpause
    else if application "Music" is running then
        tell application "Music" to playpause
    end if
end try
'''

NEXT_TRACK_SCRIPT = '''
try
    if application "Spotify" is running then
        tell application "Spotify" to next track
    else if application "Music" is running then
        tell application "Music" to next track
    end if
end try
'''


def fire(label):
    if label == "fist":
        print(f"[{time.strftime('%H:%M:%S')}] fist -> playpause")
        subprocess.run(["osascript", "-e", PLAYPAUSE_SCRIPT], check=False)
        return
    if label == "fist_double":
        print(f"[{time.strftime('%H:%M:%S')}] double fist -> next track")
        subprocess.run(["osascript", "-e", NEXT_TRACK_SCRIPT], check=False)
        return
    key_code, modifier = GESTURE_KEYS[label]
    print(f"[{time.strftime('%H:%M:%S')}] {label} -> key {key_code} {modifier}".rstrip())
    cmd = f'tell application "System Events" to key code {key_code} {modifier}'.strip()
    subprocess.run(["osascript", "-e", cmd], check=False)


def is_fist(hand):
    lm = hand.landmark
    wrist = np.array([lm[0].x, lm[0].y])
    curled = 0
    for tip_i, mcp_i in FINGER_TIP_MCP:
        tip = np.array([lm[tip_i].x, lm[tip_i].y])
        mcp = np.array([lm[mcp_i].x, lm[mcp_i].y])
        if np.linalg.norm(tip - wrist) < FIST_RATIO * np.linalg.norm(mcp - wrist):
            curled += 1
    return curled == 4


def _finger_ratio(hand, tip_i, mcp_i):
    lm = hand.landmark
    wrist = np.array([lm[0].x, lm[0].y])
    tip = np.array([lm[tip_i].x, lm[tip_i].y])
    mcp = np.array([lm[mcp_i].x, lm[mcp_i].y])
    return np.linalg.norm(tip - wrist) / max(np.linalg.norm(mcp - wrist), 1e-6)


def is_peace(hand):
    index_ext  = _finger_ratio(hand,  8,  5) > PEACE_EXTENDED_RATIO
    middle_ext = _finger_ratio(hand, 12,  9) > PEACE_EXTENDED_RATIO
    ring_curl  = _finger_ratio(hand, 16, 13) < PEACE_CURLED_RATIO
    pinky_curl = _finger_ratio(hand, 20, 17) < PEACE_CURLED_RATIO
    return index_ext and middle_ext and ring_curl and pinky_curl


def is_stationary(buffer):
    if len(buffer) < BUFFER_FRAMES:
        return False
    arr = np.array(buffer)
    return (abs(arr[-1, 0] - arr[0, 0]) < STATIONARY_DISP
            and abs(arr[-1, 1] - arr[0, 1]) < STATIONARY_DISP)


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
    fist_streak = 0
    fist_armed = True  # only fire fist on rising edge after release
    pending_fist_fire = None  # deadline to fire playpause if no second clench arrives
    peace_streak = 0
    peace_armed = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        fist_now = False
        peace_now = False
        if result.multi_hand_landmarks:
            hl = result.multi_hand_landmarks[0]
            wrist = hl.landmark[WRIST]
            buffer.append((wrist.x, wrist.y))
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
            fist_now = is_fist(hl)
            peace_now = is_peace(hl)
        else:
            buffer.clear()

        if fist_now:
            fist_streak += 1
        else:
            fist_streak = 0
            fist_armed = True

        if peace_now:
            peace_streak += 1
        else:
            peace_streak = 0
            peace_armed = True

        now = time.time()
        fist_rising = fist_armed and fist_streak >= FIST_HOLD_FRAMES

        if fist_rising:
            fist_armed = False
            if pending_fist_fire is not None:
                fire("fist_double")
                pending_fist_fire = None
                last_fired = now
                last_label = "fist_double"
                buffer.clear()
            else:
                pending_fist_fire = now + DOUBLE_FIST_WINDOW
                last_label = "fist (pending)"
        elif pending_fist_fire is not None and now >= pending_fist_fire:
            fire("fist")
            pending_fist_fire = None
            last_fired = now
            last_label = "fist"
            buffer.clear()
        elif peace_armed and peace_streak >= PEACE_HOLD_FRAMES and is_stationary(buffer):
            fire("peace")
            peace_armed = False
            last_fired = now
            last_label = "peace"
            buffer.clear()
        elif now - last_fired >= COOLDOWN_S:
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
        fist_status = f"fist:{'Y' if fist_now else 'N'} streak:{fist_streak} armed:{'Y' if fist_armed else 'N'}"
        cv2.putText(frame, f"last: {last_label}   cooldown: {cooldown_left:.2f}s   {mc_status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, fist_status,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
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

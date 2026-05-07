# hands-off

macOS hand-gesture controller for Spaces. See `Proposal.pdf` for the full plan.

## Step 1 — keystroke spike

Goal: confirm that a Python script can switch macOS Spaces via keystrokes on each teammate's machine. If this doesn't work, nothing else in the project will, so we settle it before writing any vision code.

### macOS setup checklist

Run through this once per machine. All three are required.

**1. Enable the Mission Control space-switching shortcuts.**
System Settings → Keyboard → Keyboard Shortcuts → Mission Control:
- enable **Move left a space**  (default `^ ←`)
- enable **Move right a space** (default `^ →`)

Without enabling them, no keystroke we send will move you between Spaces.

**2. Create at least two Spaces.**
Open Mission Control (F3, or three-finger swipe up) and click `+` in the top bar. With only one Space the spike has nothing to switch to.

**3. Grant your terminal app permission to drive System Events.**
The spike uses `osascript` to send keystrokes through `System Events`, which macOS reliably intercepts as Spaces shortcuts (Python's `pynput` posts events at a layer the system *doesn't* intercept — confirmed during initial testing). On first run macOS will pop a prompt: *"Terminal wants to control System Events"* — approve it.

If you click through too fast or it still doesn't fire, check both of these and add your terminal app (Terminal, iTerm, VS Code, Cursor, …) if missing:
- System Settings → Privacy & Security → **Automation** → *your terminal* → *System Events* ✓
- System Settings → Privacy & Security → **Accessibility** → *your terminal* ✓

### Run the spike

```bash
python3 scripts/spike_keystroke.py
```

The desktop should slide right, pause, then slide back left. If it doesn't, work back through the checklist above.

## Python environment (steps 2+)

MediaPipe needs Python 3.10–3.12 on macOS — 3.13 isn't supported on Apple Silicon yet.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll also need **Camera** access for whatever app you run Python from: System Settings → Privacy & Security → Camera → *your terminal* (Terminal, iTerm, VS Code, Cursor, …). macOS prompts on first webcam open; if you miss it, toggle it on here.

## Step 2 — live rule-based pipeline

`scripts/live_rules.py` is the end-to-end vertical slice: webcam → MediaPipe Hands → wrist-velocity rule → `osascript`. No trained model required. Focus the video window and press `q` to quit.

Gestures (frame is mirrored, so directions match your physical hand):

| gesture                                     | action                                   |
| ------------------------------------------- | ---------------------------------------- |
| swipe hand left                             | `Ctrl+→` — next desktop                  |
| swipe hand right                            | `Ctrl+←` — previous desktop              |
| swipe hand up                               | `Ctrl+↑` — Mission Control               |
| swipe hand down                             | `Esc` — close Mission Control            |
| fist (clench and hold ~4 frames)            | Spotify / Apple Music **play-pause**     |
| double fist (two clenches within 0.6 s)     | Spotify / Apple Music **next track**     |

A `mc_open` flag tracks Mission Control state so swipe-up doesn't refire while MC is already open, and swipe-down only fires when it is. Press `r` if the state ever gets out of sync. A 1-second cooldown after every fire prevents the same gesture from triggering repeatedly.

Tunables at the top of the file if detection feels too sensitive or too sluggish: `BUFFER_FRAMES`, `SWIPE_DX` / `SWIPE_DY` (motion thresholds), `DOMINANCE` (how strongly one axis must dominate the other), `COOLDOWN_S`, `FIST_HOLD_FRAMES`, `DOUBLE_FIST_WINDOW`.

Run:

```bash
python scripts/live_rules.py
```

## Step 3 — recording training data

`scripts/record_landmarks.py` captures 30-frame landmark windows for the eventual learned classifier. Each saved file is a `(30, 21, 3)` array — frames × landmarks × xyz — written to `data/<label>/<timestamp>.npz`.

Controls (focus the video window):

| key     | action                                                |
| ------- | ----------------------------------------------------- |
| `1`     | select label `swipe_left`                             |
| `2`     | select label `swipe_right`                            |
| `3`     | select label `swipe_up`                               |
| `4`     | select label `idle`                                   |
| `space` | 1-second countdown, then record one window            |
| `q`     | quit                                                  |

If MediaPipe loses the hand mid-window the save is aborted (HUD shows the reason). Aim for ~30+ examples per label per teammate, varied lighting / distance / angle — diversity matters more than volume.

Current state: ~30 `swipe_left` windows committed. `swipe_right`, `swipe_up`, and `idle` still need recording from all three of us.

## Up next

Train a lightweight classifier on `data/` and swap it in behind the same detection interface as the rule-based baseline, so we can A/B compare the learned approach against the hand-coded one (the comparison the proposal commits to).

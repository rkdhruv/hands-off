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

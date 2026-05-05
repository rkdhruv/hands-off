"""Verify macOS Space switching via Ctrl+Left/Right keystrokes."""

import subprocess
import time

KEY_LEFT = 123
KEY_RIGHT = 124


def tap_space(key_code, label):
    print(f"  -> {label}")
    subprocess.run(
        [
            "osascript",
            "-e",
            f'tell application "System Events" to key code {key_code} using control down',
        ],
        check=True,
    )


def main():
    print("hands-off keystroke spike")
    print("You should have at least 2 Spaces. Open Mission Control to add one.")
    print("Starting in 3 seconds...")
    for n in (3, 2, 1):
        print(f"  {n}")
        time.sleep(1)

    tap_space(KEY_RIGHT, "Ctrl+Right  (next space)")
    time.sleep(1.5)
    tap_space(KEY_LEFT, "Ctrl+Left   (previous space)")

    print()
    print("If the desktops still didn't switch, see README troubleshooting.")


if __name__ == "__main__":
    main()

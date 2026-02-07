import sys
import os
import time
import subprocess

# Config
YDOTOOL_SOCKET = "/tmp/.ydotool_socket"
YDOTOOL_BIN = "/usr/local/bin/ydotool"

def run_ydo(args):
    env = os.environ.copy()
    env["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET
    subprocess.run([YDOTOOL_BIN] + args, env=env)

def main():
    # 1. Focus input box
    # Right pane center: 1440. Y: 960.
    run_ydo(["mousemove", "-a", "-x", "1440", "-y", "960"])
    time.sleep(0.5)
    run_ydo(["click", "0x110"])
    time.sleep(1.0)
    
    # 2. Clear box (Ctrl+A, Backspace)
    # KEY_LEFTCTRL = 29, KEY_A = 30, KEY_BACKSPACE = 14
    print("Clearing...")
    run_ydo(["key", "29:1", "30:1", "30:0", "29:0"])
    time.sleep(0.5)
    run_ydo(["key", "14:1", "14:0"])
    time.sleep(0.5)
    
    # 3. Type message
    msg = "Eidos assertive. Integration verified. Ready for next phase. Check result."
    print(f"Typing: {msg}")
    run_ydo(["type", msg])
    time.sleep(1.0)
    
    # 4. Click Send Button
    # Based on analysis, let's try 1888, 965
    print("Clicking Send...")
    run_ydo(["mousemove", "-a", "-x", "1888", "-y", "965"])
    time.sleep(0.5)
    run_ydo(["click", "0x110"])
    time.sleep(0.5)
    
    # 5. Backup send (Enter)
    run_ydo(["mousemove", "-a", "-x", "1440", "-y", "960"])
    run_ydo(["click", "0x110"])
    run_ydo(["key", "28:1", "28:0"])

if __name__ == "__main__":
    main()

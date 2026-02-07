import sys
import os
sys.path.append(os.path.abspath("src"))
from computer_control_forge.control import ControlService

try:
    # Initialize with log_dir explicitly to avoid permission issues if default is bad
    # asking for specific log dir in user home
    home = os.path.expanduser("~")
    log_dir = os.path.join(home, "eidosian_forge/computer_control_forge/logs")
    
    # We must set dry_run=False to actually capture
    control = ControlService(dry_run=False, log_dir=None) 
    # Note: Context manager calls start() which might fail if safety checks fail
    # so we call start manually to debug if needed, but context manager is cleaner.
    
    control.start()
    try:
        print("Capturing...")
        img_data = control.capture_screen()
        if img_data:
            with open("current_state.png", "wb") as f:
                f.write(img_data)
            print("Saved to current_state.png")
        else:
            print("Failed to capture (returned None).")
    finally:
        control.stop()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

import sys
import os
import time
from pathlib import Path
sys.path.append(os.path.abspath("src"))
from computer_control_forge.control import ControlService

def main():
    # Targets for ChatGPT (Screen 1920x1080, Right Half 960-1920)
    # Right pane center X: 1440
    # Input box Y: ~940
    # Send button X: ~1885
    
    input_x, input_y = 1440, 940
    send_x, send_y = 1885, 940
    
    with ControlService(dry_run=False) as control:
        # 1. Perception: Initial state
        print("--- PHASE 1: INITIAL PERCEPTION ---")
        img = control.capture_screen()
        with open("gpt_step1_initial.png", "wb") as f:
            f.write(img)
        time.sleep(1.0)
        
        # 2. Focus: Click input box
        print("--- PHASE 2: FOCUSING ---")
        control.move_mouse(input_x, input_y)
        time.sleep(0.5)
        control.click(input_x, input_y)
        time.sleep(1.5) # Wait for UI to react
        
        # 3. Clear: select all and delete
        print("--- PHASE 3: CLEARING ---")
        control.press_key("ctrl+a")
        time.sleep(0.5)
        control.press_key("backspace")
        time.sleep(1.0)
        
        # 4. Perception: Verify empty focus
        print("--- PHASE 4: VERIFY FOCUS ---")
        img = control.capture_screen()
        with open("gpt_step2_focus.png", "wb") as f:
            f.write(img)
            
        # 5. Type: The message
        msg = "Eidos speaking. System audit complete. Forge integrated. Ready for Phase 3 definition. Current time: 9:15 PM."
        print(f"--- PHASE 5: TYPING ({len(msg)} chars) ---")
        control.type_text(msg, interval_ms=50)
        time.sleep(1.5)
        
        # 6. Perception: Verify text
        print("--- PHASE 6: VERIFY TEXT ---")
        img = control.capture_screen()
        with open("gpt_step3_typed.png", "wb") as f:
            f.write(img)
            
        # 7. Action: Send
        print("--- PHASE 7: SENDING ---")
        # Ensure focus
        control.move_mouse(input_x, input_y)
        control.click(input_x, input_y)
        time.sleep(0.5)
        
        # Strategy A: Tab + Enter
        print("Trying Strategy A: Tab + Enter...")
        control.press_key("tab")
        time.sleep(0.2)
        control.press_key("enter")
        time.sleep(1.0)
        
        # Strategy B: Click Send Button Grid
        print("Trying Strategy B: Button Click Grid...")
        # Coordinates from visual analysis: x ~ 1875, y ~ 970
        for x in [1870, 1875, 1880]:
            for y in [960, 970, 980]:
                control.move_mouse(x, y)
                control.click(x, y)
                time.sleep(0.1)
        
        # Strategy C: Ctrl + Enter
        print("Trying Strategy C: Ctrl+Enter...")
        control.move_mouse(input_x, input_y)
        control.click(input_x, input_y)
        time.sleep(0.2)
        control.press_key("ctrl+enter")
        
        # 8. Wait: Observe response
        print("--- PHASE 8: OBSERVING (20s) ---")
        time.sleep(20.0)
        img = control.capture_screen()
        with open("gpt_step4_final.png", "wb") as f:
            f.write(img)
        print("--- SEQUENCE COMPLETE ---")

if __name__ == "__main__":
    main()

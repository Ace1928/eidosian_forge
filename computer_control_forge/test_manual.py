#!/usr/bin/env python3
"""
Test script for computer_control_forge safety mechanisms.
"""

import sys
sys.path.insert(0, "/home/lloyd/eidosian_forge/computer_control_forge/src")

from computer_control_forge import is_kill_switch_active
from computer_control_forge.control import ControlService

def main():
    print("Testing computer_control_forge...")
    
    # Check kill switch
    print(f"Kill switch active: {is_kill_switch_active()}")
    
    # Test control service in dry-run mode
    control = ControlService(dry_run=True)
    control.start()
    print("Control service started (dry run mode)")
    
    # Test actions (dry run - no actual input)
    control.type_text("test text")
    control.click(100, 100)
    control.move_mouse(200, 200)
    
    control.stop()
    print("Control service stopped")
    
    # Check logs
    import os
    log_dir = "/home/lloyd/eidosian_forge/computer_control_forge/logs"
    logs = os.listdir(log_dir)
    print(f"Session logs created: {len(logs)}")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    main()

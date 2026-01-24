#!/usr/bin/env python3
"""
Computer Control Forge CLI

Unified CLI for Eidos computer control capabilities:
- Mouse control (human-like, precise, PID-controlled)
- Keyboard input
- Screen capture and perception
- Continuation daemon
"""

import argparse
import sys
import json
from pathlib import Path


def main(argv=None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="computer-control",
        description="🎮 Eidos Computer Control - Safe, auditable computer automation",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status
    status_parser = subparsers.add_parser("status", help="Show control system status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Safety controls
    safety_parser = subparsers.add_parser("safety", help="Safety controls")
    safety_sub = safety_parser.add_subparsers(dest="safety_cmd")
    safety_sub.add_parser("engage", help="Engage safety switch (stops all control)")
    safety_sub.add_parser("release", help="Release safety switch")
    
    # Mouse
    mouse_parser = subparsers.add_parser("mouse", help="Mouse control")
    mouse_sub = mouse_parser.add_subparsers(dest="mouse_cmd")
    
    move_parser = mouse_sub.add_parser("move", help="Move mouse to position")
    move_parser.add_argument("x", type=int, help="X coordinate")
    move_parser.add_argument("y", type=int, help="Y coordinate")
    move_parser.add_argument("--human", action="store_true", help="Use human-like movement")
    
    click_parser = mouse_sub.add_parser("click", help="Click at position")
    click_parser.add_argument("--x", type=int, help="X coordinate")
    click_parser.add_argument("--y", type=int, help="Y coordinate")
    click_parser.add_argument("--button", choices=["left", "right", "middle"], default="left")
    
    pos_parser = mouse_sub.add_parser("position", help="Show current mouse position")
    
    # Keyboard
    key_parser = subparsers.add_parser("type", help="Type text")
    key_parser.add_argument("text", help="Text to type")
    
    # Screen
    screen_parser = subparsers.add_parser("screen", help="Screen capture")
    screen_sub = screen_parser.add_subparsers(dest="screen_cmd")
    
    capture_parser = screen_sub.add_parser("capture", help="Capture screenshot")
    capture_parser.add_argument("-o", "--output", help="Output file path")
    
    # Daemon
    daemon_parser = subparsers.add_parser("daemon", help="Continuation daemon control")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_cmd")
    
    daemon_start = daemon_sub.add_parser("start", help="Start continuation daemon")
    daemon_start.add_argument("--idle-threshold", type=int, default=300, 
                              help="Seconds of idle before triggering (default 300)")
    
    daemon_stop = daemon_sub.add_parser("stop", help="Stop continuation daemon")
    daemon_status = daemon_sub.add_parser("status", help="Show daemon status")
    
    args = parser.parse_args(argv)
    
    if args.version:
        from computer_control_forge import __version__
        print(f"Computer Control Forge v{__version__}")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle commands
    try:
        if args.command == "status":
            from computer_control_forge import is_kill_switch_active, KILL_FILE, PID_FILE
            status = {
                "safety_engaged": is_kill_switch_active(),
                "safety_file": str(KILL_FILE),
                "pid_file": str(PID_FILE),
            }
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                print("🎮 Computer Control Status")
                print(f"  Safety switch: {'🛑 ENGAGED' if status['safety_engaged'] else '✓ released'}")
                print(f"  Safety file: {status['safety_file']}")
            return 0
        
        elif args.command == "safety":
            if args.safety_cmd == "engage":
                from computer_control_forge import engage_kill_switch
                engage_kill_switch()
                print("🛑 Safety switch engaged - all control stopped")
                return 0
            elif args.safety_cmd == "release":
                from computer_control_forge import KILL_FILE
                if KILL_FILE.exists():
                    KILL_FILE.unlink()
                    print("✓ Safety switch released")
                else:
                    print("Safety switch was not engaged")
                return 0
        
        elif args.command == "mouse":
            if args.mouse_cmd == "position":
                from computer_control_forge.cursor_position import get_cursor_position
                x, y = get_cursor_position()
                print(f"Cursor position: ({x}, {y})")
                return 0
            elif args.mouse_cmd == "move":
                from computer_control_forge.control import ScreenControl
                ctrl = ScreenControl()
                ctrl.move_mouse(args.x, args.y)
                print(f"Moved to ({args.x}, {args.y})")
                return 0
            elif args.mouse_cmd == "click":
                from computer_control_forge.control import ScreenControl
                ctrl = ScreenControl()
                if args.x and args.y:
                    ctrl.move_mouse(args.x, args.y)
                ctrl.click(args.button)
                print(f"Clicked {args.button}")
                return 0
        
        elif args.command == "type":
            from computer_control_forge.wayland_control import type_text
            result = type_text(args.text)
            if result.get("success"):
                print(f"Typed {len(args.text)} characters")
            else:
                print(f"Error: {result.get('error')}")
                return 1
            return 0
        
        elif args.command == "screen":
            if args.screen_cmd == "capture":
                from computer_control_forge.control import ScreenControl
                ctrl = ScreenControl()
                img = ctrl.capture_screen()
                output = args.output or "/tmp/eidos_capture.png"
                img.save(output)
                print(f"Saved screenshot to {output}")
                return 0
        
        elif args.command == "daemon":
            from computer_control_forge.daemon import (
                get_status, activate_stop_switch, deactivate_stop_switch, run_daemon
            )
            if args.daemon_cmd == "status":
                status = get_status()
                print(json.dumps(status, indent=2))
                return 0
            elif args.daemon_cmd == "stop":
                activate_stop_switch()
                print("Daemon stop requested")
                return 0
            elif args.daemon_cmd == "start":
                print(f"Starting daemon (idle_threshold={args.idle_threshold}s)")
                run_daemon(idle_threshold_sec=args.idle_threshold)
                return 0
    
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    parser.print_help()
    return 0


# Expose as callable for Typer compatibility
app = main


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
🎮 Computer Control Demonstration Patterns

Shows various mouse and keyboard control patterns using the Wayland backend.

Usage:
    python demo_patterns.py [pattern]

Patterns:
    rectangle - Mouse traces a rectangle
    circle    - Mouse traces a circle
    spiral    - Mouse traces a spiral
    wave      - Mouse traces a sine wave
    type      - Types a test message
    all       - Run all patterns

Created: 2026-01-23
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from computer_control_forge.wayland_control import (
    check_daemon, mouse_move_absolute, type_text, mouse_click
)


def demo_rectangle(width: int = 1300, height: int = 600, delay: float = 0.3):
    """Move mouse in a rectangle pattern."""
    print("📐 Rectangle pattern...")
    
    cx, cy = 960, 540
    x1, y1 = cx - width//2, cy - height//2
    x2, y2 = cx + width//2, cy + height//2
    
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    
    for x, y in corners:
        mouse_move_absolute(x, y)
        time.sleep(delay)
    
    mouse_move_absolute(cx, cy)
    print("   ✅ Complete")


def demo_circle(radius: int = 300, steps: int = 36, delay: float = 0.05):
    """Move mouse in a circle pattern."""
    print("⭕ Circle pattern...")
    
    cx, cy = 960, 540
    
    for i in range(steps + 1):
        angle = (i / steps) * 2 * math.pi
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        mouse_move_absolute(x, y)
        time.sleep(delay)
    
    mouse_move_absolute(cx, cy)
    print("   ✅ Complete")


def demo_spiral(turns: int = 3, steps_per_turn: int = 24, delay: float = 0.03):
    """Move mouse in a spiral pattern."""
    print("🌀 Spiral pattern...")
    
    cx, cy = 960, 540
    max_radius = 250
    total_steps = turns * steps_per_turn
    
    for i in range(total_steps + 1):
        angle = (i / steps_per_turn) * 2 * math.pi
        radius = (i / total_steps) * max_radius
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        mouse_move_absolute(x, y)
        time.sleep(delay)
    
    mouse_move_absolute(cx, cy)
    print("   ✅ Complete")


def demo_wave(amplitude: int = 100, waves: int = 3, steps: int = 60, delay: float = 0.02):
    """Move mouse in a sine wave pattern."""
    print("〰️ Wave pattern...")
    
    cy = 540
    start_x, end_x = 200, 1720
    
    for i in range(steps + 1):
        t = i / steps
        x = int(start_x + t * (end_x - start_x))
        y = int(cy + amplitude * math.sin(t * waves * 2 * math.pi))
        mouse_move_absolute(x, y)
        time.sleep(delay)
    
    mouse_move_absolute(960, 540)
    print("   ✅ Complete")


def demo_type(message: str = None):
    """Type a test message."""
    print("⌨️ Typing demonstration...")
    
    if message is None:
        message = "Hello! This is Eidos. "
    
    result = type_text(message)
    if result.get("success"):
        print(f"   ✅ Typed: {message[:40]}...")
    else:
        print(f"   ❌ Failed: {result.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="Computer Control Demo Patterns")
    parser.add_argument("pattern", nargs="?", default="rectangle",
                       choices=["rectangle", "circle", "spiral", "wave", "type", "all"],
                       help="Pattern to demonstrate")
    parser.add_argument("--message", "-m", default=None, help="Custom message for type demo")
    args = parser.parse_args()
    
    status = check_daemon()
    if not status.get("daemon_accessible"):
        print("❌ ydotoold daemon not accessible!")
        sys.exit(1)
    
    print("🎮 Computer Control Demo")
    print("=" * 40)
    print(f"Daemon: OK")
    print(f"Pattern: {args.pattern}\n")
    
    patterns = {
        "rectangle": demo_rectangle,
        "circle": demo_circle,
        "spiral": demo_spiral,
        "wave": demo_wave,
        "type": lambda: demo_type(args.message),
    }
    
    if args.pattern == "all":
        for name, func in patterns.items():
            func()
            time.sleep(0.5)
    else:
        patterns[args.pattern]()
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    main()

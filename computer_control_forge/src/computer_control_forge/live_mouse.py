#!/usr/bin/env python3
"""
Live Mouse Controller with Streaming Feedback

Moves mouse while streaming real-time position updates.
Designed for AI agent to observe its own actions in real-time.

Author: Eidos
Version: 1.0.0
"""

import sys
import os
import time
import json
import subprocess
import math
import random
from typing import Tuple, Optional, List
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")

from cursor_position import KWinCursorPosition

class LiveMouse:
    """Mouse controller with live streaming feedback."""
    
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    EDGE_MARGIN = 5
    
    def __init__(self):
        self.cursor = KWinCursorPosition()
        self._frame = 0
    
    def _emit(self, event_type: str, data: dict):
        """Emit JSON event."""
        event = {
            "type": event_type,
            "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "frame": self._frame,
            **data
        }
        print(json.dumps(event), flush=True)
        self._frame += 1
    
    def _move(self, dx: int, dy: int):
        """Execute relative movement."""
        subprocess.run(["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)], 
                      capture_output=True)
    
    def _click(self, button: str = "left"):
        """Click mouse button."""
        btn = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        subprocess.run(["ydotool", "click", btn.get(button, "0xC0")], capture_output=True)
    
    def _get_pos(self) -> Tuple[int, int]:
        """Get cursor position."""
        return self.cursor.get_position()
    
    def _at_edge(self, x: int, y: int) -> Optional[str]:
        """Check if at screen edge."""
        edges = []
        if x <= self.EDGE_MARGIN:
            edges.append("L")
        elif x >= self.SCREEN_WIDTH - self.EDGE_MARGIN:
            edges.append("R")
        if y <= self.EDGE_MARGIN:
            edges.append("T")
        elif y >= self.SCREEN_HEIGHT - self.EDGE_MARGIN:
            edges.append("B")
        return "".join(edges) if edges else None
    
    def move_to(self, target_x: int, target_y: int, human_like: bool = True) -> bool:
        """Move to target with live feedback."""
        
        # Clamp target
        target_x = max(self.EDGE_MARGIN, min(target_x, self.SCREEN_WIDTH - self.EDGE_MARGIN - 1))
        target_y = max(self.EDGE_MARGIN, min(target_y, self.SCREEN_HEIGHT - self.EDGE_MARGIN - 1))
        
        start_x, start_y = self._get_pos()
        total_dist = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        
        self._emit("move_start", {
            "start": [start_x, start_y],
            "target": [target_x, target_y],
            "distance": round(total_dist, 1)
        })
        
        MAX_ITERATIONS = 30
        TOLERANCE = 8
        
        for i in range(MAX_ITERATIONS):
            x, y = self._get_pos()
            
            error_x = target_x - x
            error_y = target_y - y
            dist = math.sqrt(error_x**2 + error_y**2)
            edge = self._at_edge(x, y)
            
            # Emit position update
            self._emit("pos", {
                "p": [x, y],
                "err": round(dist, 1),
                "edge": edge
            })
            
            if dist <= TOLERANCE:
                self._emit("move_done", {
                    "final": [x, y],
                    "error": round(dist, 1),
                    "iterations": i + 1,
                    "success": True
                })
                return True
            
            # Check for edge blocking
            if edge:
                if "L" in edge and error_x < 0:
                    error_x = 0
                if "R" in edge and error_x > 0:
                    error_x = 0
                if "T" in edge and error_y < 0:
                    error_y = 0
                if "B" in edge and error_y > 0:
                    error_y = 0
                
                # Recalc distance after edge constraint
                dist = math.sqrt(error_x**2 + error_y**2)
                if dist <= TOLERANCE:
                    self._emit("move_done", {
                        "final": [x, y],
                        "error": round(dist, 1),
                        "edge_stop": edge,
                        "success": True
                    })
                    return True
            
            # Proportional control with human-like gain
            gain = 0.45 if dist > 80 else 0.35 if dist > 30 else 0.3
            
            dx = int(round(error_x * gain))
            dy = int(round(error_y * gain))
            
            # Add slight curve/noise when far
            if human_like and dist > 40:
                dx += random.randint(-2, 2)
                dy += random.randint(-2, 2)
            
            # Minimum movement
            if dx == 0 and error_x != 0:
                dx = 1 if error_x > 0 else -1
            if dy == 0 and error_y != 0:
                dy = 1 if error_y > 0 else -1
            
            self._move(dx, dy)
            time.sleep(0.02)
        
        # Didn't converge
        final_x, final_y = self._get_pos()
        self._emit("move_done", {
            "final": [final_x, final_y],
            "error": round(math.sqrt((target_x-final_x)**2 + (target_y-final_y)**2), 1),
            "iterations": MAX_ITERATIONS,
            "success": False
        })
        return False
    
    def click_at(self, x: int, y: int, button: str = "left") -> bool:
        """Move to position and click."""
        success = self.move_to(x, y)
        if success:
            self._click(button)
            self._emit("click", {"pos": [x, y], "button": button})
        return success
    
    def type_text(self, text: str, wpm: int = 200):
        """Type text with live feedback."""
        self._emit("type_start", {"length": len(text), "wpm": wpm})
        
        # Calculate delay per character (wpm -> cps)
        cps = (wpm * 5) / 60  # 5 chars per word average
        delay = 1.0 / cps
        
        for i, char in enumerate(text):
            subprocess.run(["ydotool", "type", "--", char], capture_output=True)
            
            # Progress updates every 20 chars
            if (i + 1) % 20 == 0:
                self._emit("type_progress", {
                    "chars": i + 1,
                    "pct": round(100 * (i + 1) / len(text), 1)
                })
            
            time.sleep(delay + random.uniform(-0.01, 0.02))
        
        self._emit("type_done", {"total_chars": len(text)})
    
    def press_key(self, key: str):
        """Press a key."""
        key_codes = {
            "enter": "28:1 28:0",
            "tab": "15:1 15:0", 
            "escape": "1:1 1:0",
            "backspace": "14:1 14:0",
            "space": "57:1 57:0"
        }
        code = key_codes.get(key.lower(), "")
        if code:
            subprocess.run(["ydotool", "key"] + code.split(), capture_output=True)
            self._emit("key", {"key": key})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["move", "click", "type", "demo"])
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    
    mouse = LiveMouse()
    
    if args.action == "move" and args.x and args.y:
        mouse.move_to(args.x, args.y)
    elif args.action == "click" and args.x and args.y:
        mouse.click_at(args.x, args.y)
    elif args.action == "type" and args.text:
        mouse.type_text(args.text)
    elif args.action == "demo":
        # Demo sequence
        mouse._emit("demo_start", {"sequence": ["center", "corners", "type"]})
        
        # Move to center
        mouse.move_to(960, 540)
        time.sleep(0.5)
        
        # Visit corners
        for name, pos in [("TL", (100, 100)), ("TR", (1820, 100)), 
                          ("BR", (1820, 980)), ("BL", (100, 980)),
                          ("center", (960, 540))]:
            mouse._emit("demo_target", {"name": name})
            mouse.move_to(pos[0], pos[1])
            time.sleep(0.3)
        
        mouse._emit("demo_done", {})


if __name__ == "__main__":
    main()

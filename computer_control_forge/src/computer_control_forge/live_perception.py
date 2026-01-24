#!/usr/bin/env python3
"""
Live Perception Streamer

Outputs real-time perception updates as JSON lines for live feedback.
Designed to run as background process feeding updates to AI agent.

Author: Eidos
Version: 1.0.0
"""

import sys
import os
import time
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Ensure unbuffered output for live streaming
sys.stdout.reconfigure(line_buffering=True)

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")

from cursor_position import KWinCursorPosition
from perception.window_manager import WindowManager

class LivePerceptionStream:
    """Streams perception updates as JSON lines."""
    
    def __init__(self, update_hz: float = 10.0):
        self.update_hz = update_hz
        self.cursor = KWinCursorPosition()
        self.windows = WindowManager()
        
        # State tracking for deltas
        self._last_cursor = (0, 0)
        self._last_active_window = ""
        self._last_window_count = 0
        self._frame_count = 0
        self._start_time = time.time()
        
        # Screen bounds
        ws = self.windows.get_workspace_info()
        self.screen_width = ws.display_width if ws else 1920
        self.screen_height = ws.display_height if ws else 1080
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit a JSON event."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "frame": self._frame_count,
            "data": data
        }
        print(json.dumps(event), flush=True)
    
    def _check_edge(self, x: int, y: int) -> Optional[str]:
        """Check if cursor is at edge."""
        edges = []
        margin = 5
        if x <= margin:
            edges.append("left")
        elif x >= self.screen_width - margin:
            edges.append("right")
        if y <= margin:
            edges.append("top")
        elif y >= self.screen_height - margin:
            edges.append("bottom")
        return "_".join(edges) if edges else None
    
    def run(self, duration: float = 60.0):
        """Run the perception stream for duration seconds."""
        interval = 1.0 / self.update_hz
        end_time = time.time() + duration
        
        # Initial state emit
        self._emit("init", {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "update_hz": self.update_hz,
            "duration": duration
        })
        
        try:
            while time.time() < end_time:
                self._frame_count += 1
                changes = {}
                
                # Cursor position
                try:
                    x, y = self.cursor.get_position()
                    
                    # Only emit if moved significantly (>2px)
                    dx = abs(x - self._last_cursor[0])
                    dy = abs(y - self._last_cursor[1])
                    
                    if dx > 2 or dy > 2:
                        edge = self._check_edge(x, y)
                        changes["cursor"] = {
                            "x": x, "y": y,
                            "dx": x - self._last_cursor[0],
                            "dy": y - self._last_cursor[1],
                            "edge": edge
                        }
                        self._last_cursor = (x, y)
                except Exception as e:
                    changes["cursor_error"] = str(e)
                
                # Active window (check less frequently)
                if self._frame_count % 5 == 0:
                    try:
                        windows = self.windows.get_windows()
                        active = next((w for w in windows if w.active), None)
                        
                        if active and active.caption != self._last_active_window:
                            changes["active_window"] = {
                                "caption": active.caption[:60],
                                "x": active.bbox.x,
                                "y": active.bbox.y,
                                "width": active.bbox.width,
                                "height": active.bbox.height
                            }
                            self._last_active_window = active.caption
                        
                        if len(windows) != self._last_window_count:
                            changes["window_count"] = len(windows)
                            self._last_window_count = len(windows)
                    except Exception as e:
                        pass
                
                # Emit if there are changes
                if changes:
                    self._emit("delta", changes)
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            pass
        finally:
            elapsed = time.time() - self._start_time
            self._emit("end", {
                "total_frames": self._frame_count,
                "elapsed_seconds": round(elapsed, 2),
                "avg_fps": round(self._frame_count / elapsed, 1) if elapsed > 0 else 0
            })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live perception stream")
    parser.add_argument("--hz", type=float, default=10.0, help="Update rate")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds")
    args = parser.parse_args()
    
    stream = LivePerceptionStream(update_hz=args.hz)
    stream.run(duration=args.duration)


if __name__ == "__main__":
    main()

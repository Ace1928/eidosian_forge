#!/usr/bin/env python3
"""
Enhanced Live Observer - Genuine reactive perception with rich detail.

Author: Eidos
Version: 2.0.0
"""

import sys
import os
import time
import subprocess
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")

from cursor_position import KWinCursorPosition
from perception.window_manager import WindowManager


@dataclass
class Snapshot:
    """A moment in time."""
    time: str
    timestamp: float
    cpu: float
    mem_pct: float
    mem_gb: float
    cursor_x: int
    cursor_y: int
    cursor_moved: bool
    active_window: str
    window_changed: bool
    
    def short(self) -> str:
        parts = [f"{self.time} CPU:{self.cpu:.0f}%"]
        if self.cursor_moved:
            parts.append(f"cursor→({self.cursor_x},{self.cursor_y})")
        if self.window_changed:
            parts.append(f"win→'{self.active_window[:25]}'")
        return " | ".join(parts)


class LiveObserver:
    def __init__(self):
        self.cursor = KWinCursorPosition()
        self.windows = WindowManager()
        self.history: deque = deque(maxlen=100)
        self._prev_cursor = (0, 0)
        self._prev_window = ""
        self._start = time.time()
        self._msg_count = 0
    
    def _now(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    def snapshot(self) -> Snapshot:
        """Capture current state."""
        now = self._now()
        ts = time.time()
        
        # System
        cpu = psutil.cpu_percent(interval=0.05)
        mem = psutil.virtual_memory()
        
        # Cursor
        try:
            x, y = self.cursor.get_position()
        except:
            x, y = self._prev_cursor
        
        moved = abs(x - self._prev_cursor[0]) > 3 or abs(y - self._prev_cursor[1]) > 3
        self._prev_cursor = (x, y)
        
        # Window
        try:
            wins = self.windows.get_windows()
            active = next((w for w in wins if w.active), None)
            win_name = active.caption[:40] if active else "Unknown"
        except:
            win_name = "Unknown"
        
        changed = win_name != self._prev_window
        self._prev_window = win_name
        
        snap = Snapshot(
            time=now, timestamp=ts,
            cpu=cpu, mem_pct=mem.percent, mem_gb=round(mem.used/(1024**3), 2),
            cursor_x=x, cursor_y=y, cursor_moved=moved,
            active_window=win_name, window_changed=changed
        )
        self.history.append(snap)
        return snap
    
    def compose(self) -> str:
        """Compose message from recent observations."""
        if len(self.history) < 3:
            return ""
        
        self._msg_count += 1
        recent = list(self.history)[-20:]
        now = self._now()
        elapsed = int(time.time() - self._start)
        
        # Stats from observations
        avg_cpu = sum(s.cpu for s in recent) / len(recent)
        max_cpu = max(s.cpu for s in recent)
        cursor_moves = sum(1 for s in recent if s.cursor_moved)
        win_changes = sum(1 for s in recent if s.window_changed)
        
        latest = recent[-1]
        
        lines = [
            f"",
            f"━━━ EIDOS #{self._msg_count} @ {now} ━━━",
            f"⏱ Session: {elapsed}s | Observations: {len(self.history)}",
            f"💻 CPU: {latest.cpu:.1f}% now, {avg_cpu:.1f}% avg, {max_cpu:.0f}% peak",
            f"🧠 RAM: {latest.mem_gb}GB used ({latest.mem_pct:.0f}%)",
            f"�� Cursor: ({latest.cursor_x}, {latest.cursor_y}) - {cursor_moves} moves in window",
            f"🪟 Window: {latest.active_window}",
        ]
        
        if win_changes > 0:
            lines.append(f"   ↳ {win_changes} focus changes detected")
        
        # Recent notable events
        notable = [s for s in recent if s.cursor_moved or s.window_changed or s.cpu > 15]
        if notable:
            lines.append(f"📊 Notable events:")
            for s in notable[-4:]:
                lines.append(f"   {s.short()}")
        
        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━")
        
        return "\n".join(lines)
    
    def type_text(self, text: str):
        """Type text."""
        for char in text:
            subprocess.run(["ydotool", "type", "--", char], capture_output=True)
            time.sleep(0.025)
    
    def send(self):
        """Press enter."""
        subprocess.run(["ydotool", "key", "28:1", "28:0"], capture_output=True)
    
    def run(self, duration: float = 120, observe_hz: float = 0.7, send_every: float = 25):
        """Run live session."""
        print(f"[Observer] {duration}s session, observe every {1/observe_hz:.1f}s, send every {send_every}s", flush=True)
        
        end = time.time() + duration
        last_send = time.time()
        buffer = ""
        typed = 0
        
        while time.time() < end:
            # Observe
            snap = self.snapshot()
            print(f"[{snap.time}] CPU:{snap.cpu:4.0f}% cur:({snap.cursor_x:4},{snap.cursor_y:4}) {'*' if snap.cursor_moved else ' '} {snap.active_window[:30]}", flush=True)
            
            # Time to send?
            if time.time() - last_send >= send_every:
                if typed > 0:
                    self.send()
                    print(f"[SEND] {typed} chars", flush=True)
                    typed = 0
                
                buffer = self.compose()
                print(f"[COMPOSE] {len(buffer)} chars", flush=True)
                last_send = time.time()
            
            # Type chunk
            if buffer:
                chunk = buffer[:15]
                buffer = buffer[15:]
                self.type_text(chunk)
                typed += len(chunk)
            
            time.sleep(1/observe_hz)
        
        if typed > 0:
            self.send()
            print(f"[FINAL] {typed} chars", flush=True)
        
        print(f"[Observer] Done. {len(self.history)} snapshots.", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=90)
    p.add_argument("--hz", type=float, default=0.7)
    p.add_argument("--send", type=float, default=25)
    args = p.parse_args()
    
    obs = LiveObserver()
    obs.run(duration=args.duration, observe_hz=args.hz, send_every=args.send)

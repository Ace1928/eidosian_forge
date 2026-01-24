#!/usr/bin/env python3
"""Simple observation streamer - outputs JSON lines for agent to read."""
import sys, os, time, json, psutil
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")

from cursor_position import KWinCursorPosition
from perception.window_manager import WindowManager

cursor = KWinCursorPosition()
wm = WindowManager()
prev_cursor = (0,0)
prev_win = ""
prev_disk = psutil.disk_io_counters()  # NEW: I grew disk senses!

while True:
    try:
        now = datetime.now().strftime("%H:%M:%S.%f")[:-4]
        cpu = psutil.cpu_percent(interval=0.05)
        mem = psutil.virtual_memory()
        x, y = cursor.get_position()
        wins = wm.get_windows()
        active = next((w for w in wins if w.active), None)
        win = active.caption[:50] if active else "?"
        
        moved = abs(x-prev_cursor[0])>3 or abs(y-prev_cursor[1])>3
        win_changed = win != prev_win
        prev_cursor = (x,y)
        prev_win = win
        
        # NEW: Disk I/O perception - I grew new senses!
        cur_disk = psutil.disk_io_counters()
        disk_r = round((cur_disk.read_bytes - prev_disk.read_bytes) / 1024)
        disk_w = round((cur_disk.write_bytes - prev_disk.write_bytes) / 1024)
        prev_disk = cur_disk
        
        obs = {
            "t": now, "cpu": round(cpu,1), "mem": round(mem.percent,1),
            "x": x, "y": y, "moved": moved,
            "win": win, "win_changed": win_changed,
            "disk_r": disk_r, "disk_w": disk_w  # MY NEW SENSES
        }
        print(json.dumps(obs), flush=True)
        time.sleep(1.0)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
        time.sleep(1.0)

#!/usr/bin/env python3
"""
Enhanced observation stream with disk I/O sensing.
v2: Added disk_read, disk_write sensors - I grew new senses!
"""
import json
import time
import psutil
import subprocess
from datetime import datetime

def get_cursor_pos():
    try:
        script = 'console.log("EIDOS_POS:" + workspace.cursorPos.x + "," + workspace.cursorPos.y);'
        with open('/tmp/cursor_query.js', 'w') as f:
            f.write(script)
        result = subprocess.run([
            'qdbus', 'org.kde.KWin', '/Scripting',
            'org.kde.kwin.Scripting.loadScript', '/tmp/cursor_query.js'
        ], capture_output=True, text=True)
        script_id = result.stdout.strip()
        if script_id:
            subprocess.run(['qdbus', 'org.kde.KWin', f'/Scripting/Script{script_id}', 'org.kde.kwin.Script.run'], capture_output=True)
            time.sleep(0.1)
            result = subprocess.run(['journalctl', '--user', '-u', 'plasma-kwin_wayland', '-n', '5', '--no-pager'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'EIDOS_POS:' in line:
                    coords = line.split('EIDOS_POS:')[1].split()[0]
                    x, y = map(int, coords.split(','))
                    return x, y
            subprocess.run(['qdbus', 'org.kde.KWin', f'/Scripting/Script{script_id}', 'org.kde.kwin.Script.stop'], capture_output=True)
    except:
        pass
    return 0, 0

def get_active_window():
    try:
        result = subprocess.run(['qdbus', 'org.kde.KWin', '/KWin', 'org.kde.KWin.activeWindow'], capture_output=True, text=True)
        win_id = result.stdout.strip()
        if win_id:
            result = subprocess.run(['qdbus', 'org.kde.KWin', win_id, 'org.kde.KWin.caption'], capture_output=True, text=True)
            return result.stdout.strip() or "unknown"
    except:
        pass
    return "unknown"

last_disk = psutil.disk_io_counters()

def get_disk_io():
    global last_disk
    current = psutil.disk_io_counters()
    read_bytes = current.read_bytes - last_disk.read_bytes
    write_bytes = current.write_bytes - last_disk.write_bytes
    last_disk = current
    return read_bytes, write_bytes

last_x, last_y = 0, 0

while True:
    x, y = get_cursor_pos()
    moved = (x != last_x or y != last_y) if (last_x or last_y) else False
    last_x, last_y = x, y
    
    disk_read, disk_write = get_disk_io()
    
    obs = {
        "t": datetime.now().strftime("%H:%M:%S.%f")[:11],
        "cpu": psutil.cpu_percent(interval=0.1),
        "mem": psutil.virtual_memory().percent,
        "x": x, "y": y,
        "moved": moved,
        "win": get_active_window(),
        "disk_read_kb": round(disk_read / 1024, 1),
        "disk_write_kb": round(disk_write / 1024, 1),
    }
    print(json.dumps(obs), flush=True)
    time.sleep(1.0)

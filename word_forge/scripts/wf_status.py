#!/usr/bin/env python3
"""
Word Forge Status CLI.

Provides a real-time dashboard for the background daemon.
Displays uptime, processed count, error rate, and queue health.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

STATUS_FILE = "daemon_status.json"

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return f"{days}d {hours}h {mins}m {secs}s"
    if hours > 0:
        return f"{hours}h {mins}m {secs}s"
    return f"{mins}m {secs}s"

def show_status():
    path = Path(STATUS_FILE)
    if not path.exists():
        print(f"❌ Daemon status file not found: {STATUS_FILE}")
        print("Ensure the daemon is running and has processed at least one item.")
        return

    try:
        with open(path, "r") as f:
            data = json.load(f)
            
        print("\n" + "="*50)
        print(" 💎 WORD FORGE DAEMON STATUS ⚡")
        print("="*50)
        
        status_color = "🟢" if data.get("status") == "RUNNING" else "🔴"
        print(f" Status:      {status_color} {data.get('status')}")
        print(f" Uptime:      {format_duration(data.get('uptime_seconds', 0))}")
        print(f" Last Sync:   {data.get('last_heartbeat')}")
        print(f" Model:       {data.get('model', 'Unknown')}")
        
        print("\n" + "-"*50)
        print(" 📊 METRICS")
        print("-"*50)
        print(f" Processed:   {data.get('processed_items', 0)}")
        print(f" Errors:      {data.get('errors', 0)}")
        print(f" Queue Size:  {data.get('queue_size', 0)}")
        
        # Calculate throughput if uptime > 0
        uptime = data.get("uptime_seconds", 0)
        if uptime > 60:
            rate = data.get("processed_items", 0) / (uptime / 3600)
            print(f" Throughput:  {rate:.2f} items/hour")
            
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error reading status: {e}")

if __name__ == "__main__":
    try:
        while True:
            # Clear screen
            print("\033[H\033[J", end="")
            show_status()
            print("Press Ctrl+C to exit. Auto-refreshing every 5s...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nExiting.")

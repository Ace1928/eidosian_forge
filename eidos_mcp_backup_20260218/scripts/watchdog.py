#!/usr/bin/env python3
"""
Eidos MCP Watchdog
------------------
Monitors the health of the Eidos MCP server and restarts it if it becomes unresponsive.
"""
import time
import sys
import subprocess
import urllib.request
import urllib.error
from datetime import datetime

# Configuration
HEALTH_URL = "http://127.0.0.1:8928/health"
CHECK_INTERVAL = 30  # Seconds between checks
FAILURE_THRESHOLD = 3  # Number of consecutive failures before restart
SERVICE_NAME = "eidos-mcp"

def log(msg: str):
    print(f"[{datetime.now().isoformat()}] {msg}", file=sys.stdout, flush=True)

def check_health() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as response:
            if response.status == 200:
                return True
    except Exception as e:
        log(f"Health check failed: {e}")
    return False

def restart_service():
    log(f"Restarting service: {SERVICE_NAME}...")
    try:
        subprocess.run(["systemctl", "--user", "restart", SERVICE_NAME], check=True)
        log("Service restart command issued successfully.")
    except subprocess.CalledProcessError as e:
        log(f"Failed to restart service: {e}")

def main():
    log("Starting Eidos MCP Watchdog...")
    failures = 0
    
    while True:
        if check_health():
            if failures > 0:
                log(f"Service recovered after {failures} failures.")
            failures = 0
        else:
            failures += 1
            log(f"Health check failed ({failures}/{FAILURE_THRESHOLD})")
            
            if failures >= FAILURE_THRESHOLD:
                log("Failure threshold reached. Initiating restart.")
                restart_service()
                # Give it some time to come back up before checking again
                time.sleep(10)
                failures = 0
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

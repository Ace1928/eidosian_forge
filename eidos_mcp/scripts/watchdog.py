#!/usr/bin/env python3
"""
Eidos MCP Watchdog
------------------
Monitors the health of the Eidos MCP server and restarts it if it becomes unresponsive.
"""

import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parents[2]
for extra in (FORGE_ROOT / "lib", FORGE_ROOT):
    extra_str = str(extra)
    if extra.exists() and extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from eidosian_core.ports import get_service_url

# Configuration
HEALTH_URL = get_service_url("eidos_mcp", default_port=8928, default_path="/health")
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

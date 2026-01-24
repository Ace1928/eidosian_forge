#!/usr/bin/env python3
"""
🔄 Continuation Daemon Launcher

Starts the autonomous continuation daemon that monitors for idle states
and types continuation prompts to keep Eidos iterating.

Usage:
    python run_continuation_daemon.py [--idle-threshold 300] [--check-interval 30]

Stop with:
    touch /tmp/eidosian_continuation_stop

Logs at:
    /tmp/eidosian_continuation.log
"""

import argparse
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from computer_control_forge.daemon import run_daemon, get_status, deactivate_stop_switch


def main():
    parser = argparse.ArgumentParser(description="Eidos Continuation Daemon")
    parser.add_argument(
        "--idle-threshold", type=float, default=300,
        help="Seconds of idle before triggering continuation (default: 300)"
    )
    parser.add_argument(
        "--check-interval", type=float, default=30,
        help="Seconds between checks (default: 30)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show daemon status and exit"
    )
    args = parser.parse_args()
    
    if args.status:
        import json
        print(json.dumps(get_status(), indent=2))
        return
    
    print("🔄 Starting Eidos Continuation Daemon")
    print(f"   Idle threshold: {args.idle_threshold}s")
    print(f"   Check interval: {args.check_interval}s")
    print(f"   Stop with: touch /tmp/eidosian_continuation_stop")
    print()
    
    # Clear any existing stop switch
    deactivate_stop_switch()
    
    # Run daemon
    run_daemon(
        idle_threshold_sec=args.idle_threshold,
        check_interval_sec=args.check_interval
    )


if __name__ == "__main__":
    main()

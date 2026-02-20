#!/usr/bin/env python3
"""Queue manager that serializes Codex agent runs so each task starts after the previous one."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from eidosian_core import eidosian

# Setup paths and imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "eidosian_forge"))

try:
    from diagnostics_forge import DiagnosticsForge
    from gis_forge import GisCore
except ImportError:
    # Try adding the specific forge paths if the above fails
    sys.path.append(str(BASE_DIR / "eidosian_forge" / "gis_forge"))
    sys.path.append(str(BASE_DIR / "eidosian_forge" / "diagnostics_forge"))
    from diagnostics_core import DiagnosticsForge
    from gis_core import GisCore

# Import DEFAULT_QUERY from codex_run if available, else default
try:
    from codex_run import DEFAULT_QUERY
except ImportError:
    DEFAULT_QUERY = "Describe the current system status."

# Initialize Forges
# Using context/config.json as the shared configuration source
gis = GisCore(persistence_path=BASE_DIR / "context" / "config.json")
diag = DiagnosticsForge(log_dir=str(BASE_DIR / "eidosian_forge" / "logs"), service_name="codex_query")

# Configuration via GisCore with defaults
QUEUE_FILE = Path(gis.get("codex.queue_file", str(BASE_DIR / "codex_task_queue.json")))
LOCK_FILE = Path(gis.get("codex.lock_file", str(BASE_DIR / "codex_task_queue.lock")))
PYTHON_BIN = Path(gis.get("codex.python_bin", str(BASE_DIR / "eidosian_venv" / "bin" / "python3")))
RUNNER_SCRIPT = Path(__file__).resolve().parent / "codex_run.py"
POLL_INTERVAL = float(gis.get("codex.poll_interval", 1.0))


@eidosian()
@contextmanager
def queue_lock():
    """Guard queue file access with an OS-level lock."""
    # Ensure lock file exists
    if not LOCK_FILE.exists():
        LOCK_FILE.touch()

    lock_file = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


@eidosian()
def load_queue() -> list[dict]:
    """Return the current task queue, treating corrupt data as empty."""
    if not QUEUE_FILE.exists():
        return []
    try:
        return json.loads(QUEUE_FILE.read_text())
    except json.JSONDecodeError:
        diag.log_event("WARNING", f"Corrupt queue file found at {QUEUE_FILE}. resetting.")
        return []


@eidosian()
def save_queue(queue: list[dict]) -> None:
    """Persist the task queue in JSON format for visibility and recovery."""
    QUEUE_FILE.write_text(json.dumps(queue, indent=2) + "\n")


@eidosian()
def queue_entry(query: str) -> dict:
    """Create a timestamped queue entry for the given query string."""
    return {
        "id": str(uuid.uuid4()),
        "query": query,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
    }


@eidosian()
def wait_for_turn(entry_id: str) -> None:
    """Block until this entry becomes the head of the queue."""
    while True:
        with queue_lock():
            queue = load_queue()
            if queue and queue[0]["id"] == entry_id:
                return
        time.sleep(POLL_INTERVAL)


@eidosian()
def cleanup_entry(entry_id: str) -> None:
    """Remove the completed entry from the queue so the next task can proceed."""
    with queue_lock():
        queue = load_queue()
        queue = [item for item in queue if item["id"] != entry_id]
        save_queue(queue)


@eidosian()
def ensure_runner() -> None:
    """Make sure the configured Python interpreter and runner script exist."""
    if not PYTHON_BIN.exists():
        error_msg = f"{PYTHON_BIN} bin not found; activate eidosian_venv first."
        diag.log_event("ERROR", error_msg)
        raise FileNotFoundError(error_msg)
    if not RUNNER_SCRIPT.exists():
        error_msg = f"{RUNNER_SCRIPT} is missing; cannot launch the agent."
        diag.log_event("ERROR", error_msg)
        raise FileNotFoundError(error_msg)


@eidosian()
def launch_task(entry: dict) -> None:
    """Invoke the codex runner via the dedicated virtual environment."""
    ensure_runner()
    diag.log_event("INFO", f"Launching task {entry['id']}")
    try:
        # We allow stdout/stderr to stream to console so the user sees progress
        subprocess.run(
            [str(PYTHON_BIN), str(RUNNER_SCRIPT), entry["query"]],
            check=True,
        )
        diag.log_event("INFO", f"Task {entry['id']} finished successfully.")
    except subprocess.CalledProcessError as e:
        diag.log_event("ERROR", f"Task {entry['id']} failed with exit code {e.returncode}")
        raise
    finally:
        cleanup_entry(entry["id"])


@eidosian()
def main() -> None:
    parser = argparse.ArgumentParser(description="Queue and run codex agent queries one at a time.")
    parser.add_argument("query", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    user_query = " ".join(args.query).strip()
    if not user_query:
        user_query = DEFAULT_QUERY

    entry = queue_entry(user_query)
    with queue_lock():
        queue = load_queue()
        queue.append(entry)
        save_queue(queue)

    diag.log_event("INFO", f"Queued codex task {entry['id']}. Waiting for turn...")
    wait_for_turn(entry["id"])

    diag.log_event("INFO", f"Running codex task {entry['id']}", query=entry["query"])
    launch_task(entry)
    diag.log_event("INFO", f"Codex task {entry['id']} completed.")


if __name__ == "__main__":
    main()

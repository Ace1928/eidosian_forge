"""
⚙️ Task Automation Plugin for MCP

Provides tools for scheduling, queuing, and automating tasks.
Enables persistent task management and execution tracking.

Created: 2026-01-23
"""

from __future__ import annotations
from eidosian_core import eidosian

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional

PLUGIN_MANIFEST = {
    "id": "task_automation",
    "name": "Task Automation",
    "version": "1.0.0",
    "description": "Task scheduling, queuing, and execution tracking",
    "author": "Eidos",
    "tools": [
        "task_queue_add",
        "task_queue_list",
        "task_queue_status",
        "task_execute",
        "task_schedule",
        "task_cancel"
    ]
}

# Task storage
TASK_DIR = Path("/home/lloyd/eidosian_forge/data/tasks")
TASK_DIR.mkdir(parents=True, exist_ok=True)

# In-memory task queue
_task_queue: Queue = Queue()
_running_tasks: Dict[str, Dict[str, Any]] = {}
_completed_tasks: Dict[str, Dict[str, Any]] = {}
_scheduled_tasks: Dict[str, Dict[str, Any]] = {}


def _generate_task_id() -> str:
    """Generate a unique task ID."""
    return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _save_task(task_id: str, task_data: Dict[str, Any]) -> Path:
    """Save task data to disk."""
    path = TASK_DIR / f"{task_id}.json"
    with open(path, 'w') as f:
        json.dump(task_data, f, indent=2)
    return path


def _load_tasks() -> List[Dict[str, Any]]:
    """Load all tasks from disk."""
    tasks = []
    for task_file in TASK_DIR.glob("task_*.json"):
        try:
            with open(task_file) as f:
                tasks.append(json.load(f))
        except Exception:
            pass
    return tasks


@eidosian()
def task_queue_add(
    command: str,
    description: str = "",
    priority: int = 5,
    timeout: int = 300,
    working_dir: Optional[str] = None
) -> str:
    """
    Add a task to the execution queue.
    
    Args:
        command: Shell command to execute
        description: Human-readable description
        priority: Priority 1-10 (1 = highest)
        timeout: Timeout in seconds
        working_dir: Working directory for execution
    
    Returns:
        JSON string with task info
    """
    task_id = _generate_task_id()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    task = {
        "id": task_id,
        "command": command,
        "description": description,
        "priority": priority,
        "timeout": timeout,
        "working_dir": working_dir or str(Path.home()),
        "status": "queued",
        "created_at": timestamp,
        "started_at": None,
        "completed_at": None,
        "output": None,
        "error": None,
        "exit_code": None
    }
    
    _task_queue.put((priority, task_id, task))
    _save_task(task_id, task)
    
    return json.dumps({
        "status": "success",
        "task_id": task_id,
        "position_in_queue": _task_queue.qsize(),
        "priority": priority,
        "timestamp": timestamp
    })


@eidosian()
def task_queue_list(
    status: Optional[str] = None,
    limit: int = 50
) -> str:
    """
    List tasks in the queue.
    
    Args:
        status: Filter by status (queued, running, completed, failed)
        limit: Maximum number of tasks to return
    
    Returns:
        JSON string with task list
    """
    tasks = _load_tasks()
    
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    
    # Sort by created_at descending
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    tasks = tasks[:limit]
    
    return json.dumps({
        "status": "success",
        "count": len(tasks),
        "tasks": [
            {
                "id": t["id"],
                "description": t.get("description", "")[:50],
                "status": t.get("status"),
                "priority": t.get("priority"),
                "created_at": t.get("created_at"),
                "exit_code": t.get("exit_code")
            }
            for t in tasks
        ],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, indent=2)


@eidosian()
def task_queue_status() -> str:
    """
    Get overall task queue status.
    
    Returns:
        JSON string with queue statistics
    """
    tasks = _load_tasks()
    
    status_counts = {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0
    }
    
    for task in tasks:
        status = task.get("status", "unknown")
        if status in status_counts:
            status_counts[status] += 1
    
    return json.dumps({
        "status": "success",
        "queue_size": _task_queue.qsize(),
        "running_count": len(_running_tasks),
        "status_counts": status_counts,
        "total_tasks": len(tasks),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@eidosian()
def task_execute(task_id: str) -> str:
    """
    Execute a specific task immediately.
    
    Args:
        task_id: ID of the task to execute
    
    Returns:
        JSON string with execution result
    """
    task_path = TASK_DIR / f"{task_id}.json"
    
    if not task_path.exists():
        return json.dumps({"status": "error", "error": f"Task {task_id} not found"})
    
    with open(task_path) as f:
        task = json.load(f)
    
    if task.get("status") == "running":
        return json.dumps({"status": "error", "error": "Task already running"})
    
    # Update status
    task["status"] = "running"
    task["started_at"] = datetime.now(timezone.utc).isoformat()
    _running_tasks[task_id] = task
    _save_task(task_id, task)
    
    try:
        # Execute command
        result = subprocess.run(
            task["command"],
            shell=True,
            capture_output=True,
            text=True,
            timeout=task.get("timeout", 300),
            cwd=task.get("working_dir")
        )
        
        task["status"] = "completed" if result.returncode == 0 else "failed"
        task["output"] = result.stdout[:50000] if result.stdout else ""
        task["error"] = result.stderr[:10000] if result.stderr else ""
        task["exit_code"] = result.returncode
        
    except subprocess.TimeoutExpired:
        task["status"] = "failed"
        task["error"] = f"Task timed out after {task.get('timeout')} seconds"
        task["exit_code"] = -1
        
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)
        task["exit_code"] = -1
    
    task["completed_at"] = datetime.now(timezone.utc).isoformat()
    del _running_tasks[task_id]
    _completed_tasks[task_id] = task
    _save_task(task_id, task)
    
    return json.dumps({
        "status": "success",
        "task_id": task_id,
        "task_status": task["status"],
        "exit_code": task["exit_code"],
        "output_preview": task["output"][:500] if task["output"] else "",
        "error_preview": task["error"][:500] if task["error"] else "",
        "duration_seconds": None,  # Could calculate from started_at/completed_at
        "timestamp": task["completed_at"]
    }, indent=2)


@eidosian()
def task_schedule(
    command: str,
    delay_seconds: int,
    description: str = "",
    repeat: bool = False
) -> str:
    """
    Schedule a task to run after a delay.
    
    Args:
        command: Shell command to execute
        delay_seconds: Seconds to wait before execution
        description: Human-readable description
        repeat: If True, repeat after each execution
    
    Returns:
        JSON string with schedule info
    """
    task_id = _generate_task_id()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    scheduled_time = datetime.now(timezone.utc).timestamp() + delay_seconds
    
    task = {
        "id": task_id,
        "command": command,
        "description": description,
        "scheduled_for": datetime.fromtimestamp(scheduled_time, timezone.utc).isoformat(),
        "delay_seconds": delay_seconds,
        "repeat": repeat,
        "status": "scheduled",
        "created_at": timestamp
    }
    
    _scheduled_tasks[task_id] = task
    _save_task(task_id, task)
    
    return json.dumps({
        "status": "success",
        "task_id": task_id,
        "scheduled_for": task["scheduled_for"],
        "delay_seconds": delay_seconds,
        "repeat": repeat,
        "timestamp": timestamp
    })


@eidosian()
def task_cancel(task_id: str) -> str:
    """
    Cancel a queued or scheduled task.
    
    Args:
        task_id: ID of the task to cancel
    
    Returns:
        JSON string with result
    """
    task_path = TASK_DIR / f"{task_id}.json"
    
    if not task_path.exists():
        return json.dumps({"status": "error", "error": f"Task {task_id} not found"})
    
    with open(task_path) as f:
        task = json.load(f)
    
    if task.get("status") == "running":
        return json.dumps({"status": "error", "error": "Cannot cancel running task"})
    
    if task.get("status") in ("completed", "failed"):
        return json.dumps({"status": "error", "error": "Task already finished"})
    
    task["status"] = "cancelled"
    task["cancelled_at"] = datetime.now(timezone.utc).isoformat()
    _save_task(task_id, task)
    
    # Remove from scheduled tasks if present
    if task_id in _scheduled_tasks:
        del _scheduled_tasks[task_id]
    
    return json.dumps({
        "status": "success",
        "task_id": task_id,
        "previous_status": task.get("status"),
        "timestamp": task["cancelled_at"]
    })

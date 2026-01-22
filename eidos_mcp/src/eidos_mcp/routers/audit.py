from __future__ import annotations

from ..core import tool
from ..state import audit


@tool(
    name="audit_add_todo",
    description="Append a TODO item to the system TODO list.",
    parameters={
        "type": "object",
        "properties": {
            "section": {"type": "string"},
            "task_text": {"type": "string"},
            "task_id": {"type": "string"},
        },
        "required": ["section", "task_text"],
    },
)
def audit_add_todo(section: str, task_text: str, task_id: str | None = None) -> str:
    """Append a TODO item to the system TODO list."""
    if not audit:
        return "Error: Audit forge unavailable"
    added = audit.todo_manager.add_task(section, task_text, task_id=task_id)
    return "Added" if added else "No-op: Task already exists"


@tool(
    name="audit_mark_reviewed",
    description="Mark a path as reviewed in the audit coverage map.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "agent_id": {"type": "string"},
            "scope": {"type": "string"},
        },
        "required": ["path", "agent_id"],
    },
)
def audit_mark_reviewed(path: str, agent_id: str, scope: str = "shallow") -> str:
    """Mark a path as reviewed in the audit coverage map."""
    if not audit:
        return "Error: Audit forge unavailable"
    audit.coverage.mark_reviewed(path, agent_id=agent_id, scope=scope)
    return f"Marked reviewed: {path}"

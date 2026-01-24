from __future__ import annotations

from ..core import tool
from ..state import audit, ROOT_DIR, FORGE_DIR
from ..transactions import begin_transaction
from pathlib import Path
from eidosian_core import eidosian


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
@eidosian()
def audit_add_todo(section: str, task_text: str, task_id: str | None = None) -> str:
    """Append a TODO item to the system TODO list."""
    if not audit:
        return "Error: Audit forge unavailable"
    
    # Audit modifies TODO.md
    todo_path = ROOT_DIR / "TODO.md"
    txn = begin_transaction("audit_add_todo", [todo_path])
    
    try:
        added = audit.todo_manager.add_task(section, task_text, task_id=task_id)
        if not added:
            txn.rollback("no-op: task already exists")
            return "No-op: Task already exists"
        txn.commit()
        return f"Added ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error: {exc}"


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
@eidosian()
def audit_mark_reviewed(path: str, agent_id: str, scope: str = "shallow") -> str:
    """Mark a path as reviewed in the audit coverage map."""
    if not audit:
        return "Error: Audit forge unavailable"
    
    # Audit modifies coverage_map.json in audit_data
    coverage_path = FORGE_DIR / "audit_data" / "coverage_map.json"
    txn = begin_transaction("audit_mark_reviewed", [coverage_path])
    
    try:
        audit.coverage.mark_reviewed(path, agent_id=agent_id, scope=scope)
        txn.commit()
        return f"Marked reviewed: {path} ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error: {exc}"

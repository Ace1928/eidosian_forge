from __future__ import annotations

import json
from typing import Any, Dict

from ..core import tool
from ..state import type_forge
from ..transactions import begin_transaction, find_latest_transaction_for_path, load_transaction
from pathlib import Path
import os
from eidosian_core import eidosian


_TYPE_SNAPSHOT_PATH = Path(
    os.environ.get("EIDOS_TYPE_SNAPSHOT_PATH", "~/.eidosian/type_forge_snapshot.json")
).expanduser()


def _ensure_type_persistence():
    """Ensure in-memory types are saved to disk."""
    if not type_forge:
        return
    _TYPE_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TYPE_SNAPSHOT_PATH.write_text(
        json.dumps(type_forge.snapshot(), indent=2), encoding="utf-8"
    )


@tool(
    name="type_register",
    description="Register or update a schema in Type Forge.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "schema": {"type": "object"},
        },
        "required": ["name", "schema"],
    },
)
@eidosian()
def type_register(name: str, schema: Dict[str, Any]) -> str:
    """Register or update a schema in Type Forge."""
    if not type_forge:
        return "Error: Type forge unavailable"
    
    with begin_transaction("type_register", [_TYPE_SNAPSHOT_PATH]) as txn:
        changed = type_forge.register_schema(name, schema)
        if changed:
            _ensure_type_persistence()
            return f"Updated ({txn.id})"
        txn.rollback("no-op: unchanged")
        return "No-op: Schema unchanged"


@tool(
    name="type_validate",
    description="Validate data against a registered schema.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "data": {"type": "object"},
        },
        "required": ["name", "data"],
    },
)
@eidosian()
def type_validate(name: str, data: Dict[str, Any]) -> str:
    """Validate data against a registered schema."""
    if not type_forge:
        return "Error: Type forge unavailable"
    try:
        type_forge.validate(name, data)
        return "valid"
    except Exception as exc:
        return f"invalid: {exc}"


@tool(
    name="type_snapshot",
    description="Create a snapshot of registered schemas.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def type_snapshot() -> str:
    """Snapshot registered schemas to disk."""
    if not type_forge:
        return "Error: Type forge unavailable"
    with begin_transaction("type_snapshot", [_TYPE_SNAPSHOT_PATH]) as txn:
        _ensure_type_persistence()
        return f"Snapshot created ({txn.id})"


@tool(
    name="type_restore_snapshot",
    description="Restore registered schemas from the latest snapshot.",
    parameters={
        "type": "object",
        "properties": {"transaction_id": {"type": "string"}},
    },
)
@eidosian()
def type_restore_snapshot(transaction_id: str | None = None) -> str:
    """Restore registered schemas from a snapshot transaction."""
    if not type_forge:
        return "Error: Type forge unavailable"
    txn_id = transaction_id or find_latest_transaction_for_path(_TYPE_SNAPSHOT_PATH)
    if not txn_id:
        return "Error: No transaction found for type snapshot"
    txn = load_transaction(txn_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("type_restore_snapshot")
    if _TYPE_SNAPSHOT_PATH.exists():
        snapshot = json.loads(_TYPE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        type_forge.restore(snapshot)
    return f"Type schemas restored ({txn_id})"
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..core import tool
from ..state import gis, GIS_PATH
from ..transactions import begin_transaction, find_latest_transaction_for_path, load_transaction


@tool(
    name="gis_get",
    description="Retrieve a configuration value from GIS.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "default": {"type": "string"},
        },
        "required": ["key"],
    },
)
def gis_get(key: str, default: Optional[str] = None) -> str:
    """Retrieve a configuration value from GIS."""
    if not gis:
        return "Error: GIS unavailable"
    value = gis.get(key, default=default)
    return json.dumps(value)


@tool(
    name="gis_set",
    description="Set a configuration value in GIS.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["key", "value"],
    },
)
def gis_set(key: str, value: Any) -> str:
    """Set a configuration value in GIS."""
    if not gis:
        return "Error: GIS unavailable"
    with begin_transaction("gis_set", [Path(GIS_PATH)]) as txn:
        gis.set(key, value)
        return f"GIS updated ({txn.id})"


@tool(
    name="gis_snapshot",
    description="Create a snapshot of the GIS persistence store.",
    parameters={"type": "object", "properties": {}},
)
def gis_snapshot() -> str:
    """Snapshot GIS persistence store."""
    if not gis:
        return "Error: GIS unavailable"
    txn = begin_transaction("gis_snapshot", [Path(GIS_PATH)])
    txn.commit("snapshot")
    return f"Snapshot created ({txn.id})"


@tool(
    name="gis_restore",
    description="Restore the GIS persistence store from a snapshot.",
    parameters={
        "type": "object",
        "properties": {"transaction_id": {"type": "string"}},
    },
)
def gis_restore(transaction_id: Optional[str] = None) -> str:
    """Restore GIS persistence store from a snapshot transaction."""
    if not gis:
        return "Error: GIS unavailable"
    txn_id = transaction_id or find_latest_transaction_for_path(Path(GIS_PATH))
    if not txn_id:
        return "Error: No transaction found for GIS"
    txn = load_transaction(txn_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("gis_restore")
    try:
        gis.load(GIS_PATH)
    except Exception:
        pass
    return f"GIS restored ({txn_id})"

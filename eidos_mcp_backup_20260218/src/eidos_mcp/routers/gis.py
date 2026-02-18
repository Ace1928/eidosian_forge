from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..core import tool
from ..state import gis, GIS_PATH, _is_valid_json
from ..transactions import begin_transaction, find_latest_transaction_for_path, load_transaction
from eidosian_core import eidosian


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
@eidosian()
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
@eidosian()
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
@eidosian()
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
@eidosian()
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

    # Reload from the best available persistence file without throwing noisy decode errors.
    primary = Path(GIS_PATH)
    fallback = Path.home() / ".eidosian" / "gis_data.local.json"
    reload_path = primary if _is_valid_json(primary) else fallback
    if _is_valid_json(reload_path):
        try:
            gis.load(reload_path)
        except Exception:
            pass

    return f"GIS restored ({txn_id})"

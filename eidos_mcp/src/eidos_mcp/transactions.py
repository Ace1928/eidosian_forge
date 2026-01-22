from __future__ import annotations

import functools
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

# Define a base directory for storing transaction snapshots
EIDOS_TXN_DIR = Path(
    os.environ.get("EIDOS_TXN_DIR", "~/.eidosian/transactions")
).expanduser()
EIDOS_TXN_DIR.mkdir(parents=True, exist_ok=True)

# Define a base directory for storing idempotency records
EIDOS_IDEMPOTENCY_DIR = Path(
    os.environ.get("EIDOS_IDEMPOTENCY_DIR", "~/.eidosian/idempotency")
).expanduser()
EIDOS_IDEMPOTENCY_DIR.mkdir(parents=True, exist_ok=True)


class Transaction:
    """
    Represents a single transactional operation, capable of snapshotting
    changes and rolling back.
    """

    def __init__(self, txn_id: str, action: str, paths: List[Path], create_snapshot: bool = True):
        self.id = txn_id
        self.action = action
        self.timestamp = datetime.now().isoformat()
        self.paths = [p.resolve() if p else None for p in paths if p]
        self.paths = [p for p in self.paths if p] # Filter out None
        self.snapshots: Dict[Path, Optional[Path]] = {}  # original_path: snapshot_path
        self.status = "PENDING"  # PENDING, COMMITTED, ROLLED_BACK, FAILED
        self.error_reason: Optional[str] = None
        
        if create_snapshot:
            # Save pending state immediately so we track the attempt
            _save_transaction(self)
            try:
                self._create_snapshot()
                _save_transaction(self) # Update with snapshot info
            except Exception as e:
                self.status = "FAILED"
                self.error_reason = f"Snapshot failed: {e}"
                _save_transaction(self)
                raise

    def _create_snapshot(self):
        """Creates snapshots of the current state of the paths."""
        txn_snapshot_dir = EIDOS_TXN_DIR / self.id
        txn_snapshot_dir.mkdir(parents=True, exist_ok=True)

        for path in self.paths:
            if path.exists():
                # We use a unique name in the snapshot dir to avoid collisions if multiple paths have same name
                rel_id = str(uuid.uuid4())[:8]
                snapshot_file = txn_snapshot_dir / f"{rel_id}_{path.name}"
                if path.is_file():
                    shutil.copy2(path, snapshot_file)
                    self.snapshots[path] = snapshot_file
                elif path.is_dir():
                    shutil.copytree(path, snapshot_file)
                    self.snapshots[path] = snapshot_file
            else:
                self.snapshots[path] = None  # Record that it didn't exist

    def commit(self, reason: Optional[str] = None):
        """Marks the transaction as committed."""
        self.status = "COMMITTED"
        if reason:
            self.error_reason = reason
        _save_transaction(self)

    def rollback(self, reason: str = "unknown"):
        """Restores the paths to their snapshot state."""
        rollback_errors = []
        for original_path, snapshot_path in self.snapshots.items():
            try:
                if original_path.exists():
                    if original_path.is_file():
                        original_path.unlink()
                    elif original_path.is_dir():
                        shutil.rmtree(original_path)

                if snapshot_path and snapshot_path.exists():
                    if snapshot_path.is_file():
                        shutil.copy2(snapshot_path, original_path)
                    elif snapshot_path.is_dir():
                        shutil.copytree(snapshot_path, original_path)
            except Exception as e:
                rollback_errors.append(f"{original_path}: {e}")
        
        self.status = "ROLLED_BACK"
        if rollback_errors:
            self.status = "PARTIAL_ROLLBACK"
            self.error_reason = f"Rollback reason: {reason}. Errors: {'; '.join(rollback_errors)}"
        else:
            self.error_reason = reason
        _save_transaction(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback(f"Exception: {exc_val}")
        elif self.status == "PENDING":
            self.commit()


def _save_transaction(txn: Transaction):
    """Saves transaction metadata to a JSON file."""
    txn_file = EIDOS_TXN_DIR / f"{txn.id}.json"
    data = {
        "id": txn.id,
        "action": txn.action,
        "timestamp": txn.timestamp,
        "paths": [str(p) for p in txn.paths],
        "snapshots": {str(orig): str(snap) if snap else None for orig, snap in txn.snapshots.items()},
        "status": txn.status,
        "error_reason": txn.error_reason,
    }
    txn_file.write_text(json.dumps(data, indent=2))


def load_transaction(txn_id: str) -> Optional[Transaction]:
    """Loads a transaction from its metadata file."""
    txn_file = EIDOS_TXN_DIR / f"{txn_id}.json"
    if not txn_file.exists():
        return None
    data = json.loads(txn_file.read_text())
    txn = Transaction(data["id"], data["action"], [Path(p) for p in data.get("paths", [])], create_snapshot=False)
    txn.timestamp = data.get("timestamp", txn.timestamp)
    txn.snapshots = {
        Path(orig): Path(snap) if snap else None
        for orig, snap in data.get("snapshots", {}).items()
    }
    txn.status = data.get("status", txn.status)
    txn.error_reason = data.get("error_reason")
    return txn


def begin_transaction(action: str, paths: List[Path]) -> Transaction:
    """Starts a new transactional operation."""
    txn_id = str(uuid.uuid4())
    return Transaction(txn_id, action, paths)


def list_transactions(limit: Optional[int] = 50) -> List[Dict[str, Any]]:
    """Lists recent transactions."""
    transactions = []
    for txn_file in EIDOS_TXN_DIR.glob("*.json"):
        try:
            data = json.loads(txn_file.read_text())
            if "timestamp" not in data:
                continue
            transactions.append(data)
        except json.JSONDecodeError:
            pass
    transactions.sort(key=lambda x: x["timestamp"], reverse=True)
    if limit is None:
        return transactions
    return transactions[:limit]


def find_latest_transaction_for_path(path: Path) -> Optional[str]:
    """Finds the latest transaction that involved a specific path."""
    target_path_str = str(path.resolve())
    latest_txn_id = None
    latest_timestamp = None

    for txn_data in list_transactions(limit=None):
        if target_path_str in [str(Path(p).resolve()) for p in txn_data.get("paths", [])]:
            txn_timestamp = datetime.fromisoformat(txn_data["timestamp"])
            if latest_timestamp is None or txn_timestamp > latest_timestamp:
                latest_timestamp = txn_timestamp
                latest_txn_id = txn_data["id"]
    return latest_txn_id


def rollback_all(reason: str = "emergency_reset"):
    """Rolls back all recent PENDING or COMMITTED transactions in reverse order."""
    txns = list_transactions(limit=None)
    for txn_data in txns:
        if txn_data["status"] in ["PENDING", "COMMITTED"]:
            txn = load_transaction(txn_data["id"])
            if txn:
                txn.rollback(reason)


def hash_paths(paths: List[Path]) -> str:
    """Generates a hash of the content of the given paths."""
    details = []
    for path in paths:
        if path.exists():
            details.append(f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime}")
        else:
            details.append(f"{path.name}:NON_EXISTENT")
    return str(hash(tuple(details)))


def check_idempotency(idempotency_key: str, command: str, target_state_hash: str) -> bool:
    """Checks if a command has been run idempotently."""
    record_file = EIDOS_IDEMPOTENCY_DIR / f"{idempotency_key}.json"
    if record_file.exists():
        try:
            record = json.loads(record_file.read_text())
            if record.get("command") == command and record.get("target_state_hash") == target_state_hash:
                return True
        except json.JSONDecodeError:
            pass
    return False


def record_idempotency(idempotency_key: str, command: str, target_state_hash: str):
    """Records an idempotent operation."""
    record_file = EIDOS_IDEMPOTENCY_DIR / f"{idempotency_key}.json"
    record = {
        "command": command,
        "target_state_hash": target_state_hash,
        "timestamp": datetime.now().isoformat(),
    }
    record_file.write_text(json.dumps(record, indent=2))


def transactional(action_name: str, get_paths_func: Any):
    """
    Decorator for tools to automatically wrap them in a transaction.
    get_paths_func should take the same arguments as the tool and return List[Path].
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                paths = get_paths_func(*args, **kwargs)
            except Exception:
                paths = []
                
            with begin_transaction(action_name, paths) as txn:
                return func(*args, **kwargs)
        return wrapper
    return decorator
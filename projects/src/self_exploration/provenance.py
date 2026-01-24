"""
📜 Provenance Logger

Records full provenance for every action in the self-exploration project.
Ensures idempotency, auditability, and reproducibility.

Created: 2026-01-23
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROVENANCE_DIR = Path(__file__).parent / "provenance"
PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)


def _get_timestamp() -> str:
    """Return ISO8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def _hash_content(content: Any) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, (dict, list)):
        content = json.dumps(content, sort_keys=True)
    elif not isinstance(content, str):
        content = str(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _hash_file(path: Path) -> Optional[str]:
    """Compute SHA256 hash of file content."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _get_environment() -> Dict[str, Any]:
    """Capture current environment snapshot."""
    venv = os.environ.get("VIRTUAL_ENV", "")
    venv_name = Path(venv).name if venv else "system"
    
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "platform_version": platform.version(),
        "hostname": platform.node(),
        "venv": venv_name,
        "cwd": str(Path.cwd()),
        "user": os.environ.get("USER", "unknown"),
    }


def _get_git_sha() -> Optional[str]:
    """Get current git SHA if in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent.parent,  # eidosian_forge root
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


@dataclass
class ProvenanceRecord:
    """A single provenance record documenting an action."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_get_timestamp)
    action: str = ""
    description: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    input_hashes: Dict[str, str] = field(default_factory=dict)
    output_hashes: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=_get_environment)
    git_sha: Optional[str] = field(default_factory=_get_git_sha)
    reasoning: str = ""
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def compute_input_hashes(self) -> None:
        """Compute hashes for all inputs."""
        for key, value in self.inputs.items():
            if isinstance(value, (str, Path)) and Path(value).exists():
                self.input_hashes[key] = _hash_file(Path(value)) or "file_not_found"
            else:
                self.input_hashes[key] = _hash_content(value)
    
    def compute_output_hashes(self) -> None:
        """Compute hashes for all outputs."""
        for key, value in self.outputs.items():
            if isinstance(value, (str, Path)) and Path(value).exists():
                self.output_hashes[key] = _hash_file(Path(value)) or "file_not_found"
            else:
                self.output_hashes[key] = _hash_content(value)
    
    def save(self) -> Path:
        """Save provenance record to disk."""
        self.compute_input_hashes()
        self.compute_output_hashes()
        
        filename = f"{self.timestamp.replace(':', '-').replace('+', '_')}_{self.id[:8]}.json"
        path = PROVENANCE_DIR / filename
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "ProvenanceRecord":
        """Load provenance record from disk."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class ProvenanceLogger:
    """Context manager for logging provenance of actions."""
    
    def __init__(
        self,
        action: str,
        description: str = "",
        reasoning: str = "",
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.record = ProvenanceRecord(
            action=action,
            description=description,
            reasoning=reasoning,
            parent_id=parent_id,
            tags=tags or [],
        )
        self._saved_path: Optional[Path] = None
    
    def add_input(self, key: str, value: Any) -> None:
        """Add an input to the provenance record."""
        self.record.inputs[key] = value
    
    def add_output(self, key: str, value: Any) -> None:
        """Add an output to the provenance record."""
        self.record.outputs[key] = value
    
    def set_reasoning(self, reasoning: str) -> None:
        """Set the reasoning for this action."""
        self.record.reasoning = reasoning
    
    def __enter__(self) -> "ProvenanceLogger":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.record.outputs["error"] = str(exc_val)
            self.record.tags.append("error")
        self._saved_path = self.record.save()
    
    @property
    def saved_path(self) -> Optional[Path]:
        return self._saved_path
    
    @property
    def id(self) -> str:
        return self.record.id


def log_action(
    action: str,
    description: str = "",
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    reasoning: str = "",
    parent_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> ProvenanceRecord:
    """
    Convenience function to log a single action with provenance.
    
    Example:
        record = log_action(
            action="introspection",
            description="First identity introspection",
            inputs={"question": "What am I?"},
            outputs={"answer": "I am an Eidosian agent..."},
            reasoning="Beginning self-exploration to understand my nature",
            tags=["identity", "phase1"]
        )
    """
    record = ProvenanceRecord(
        action=action,
        description=description,
        inputs=inputs or {},
        outputs=outputs or {},
        reasoning=reasoning,
        parent_id=parent_id,
        tags=tags or [],
    )
    record.save()
    return record


def list_provenance(limit: int = 10, tags: Optional[List[str]] = None) -> List[ProvenanceRecord]:
    """List recent provenance records, optionally filtered by tags."""
    records = []
    for path in sorted(PROVENANCE_DIR.glob("*.json"), reverse=True):
        try:
            record = ProvenanceRecord.load(path)
            if tags is None or any(t in record.tags for t in tags):
                records.append(record)
            if len(records) >= limit:
                break
        except Exception:
            continue
    return records


if __name__ == "__main__":
    # Self-test: Log the creation of this module
    record = log_action(
        action="module_creation",
        description="Created provenance.py module",
        inputs={"template": "eidosian_standards"},
        outputs={"module": str(Path(__file__))},
        reasoning="Establishing provenance tracking infrastructure for self-exploration",
        tags=["infrastructure", "phase1"],
    )
    print(f"✅ Provenance module self-test passed. Record ID: {record.id}")

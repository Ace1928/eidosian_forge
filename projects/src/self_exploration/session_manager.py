"""
💾 Session State Persistence for Eidosian MCP

Maintains state across sessions including:
- Plugin metrics
- Cache state
- Task queue
- Identity checkpoints

Created: 2026-01-23
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add MCP to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "eidos_mcp" / "src"))


@dataclass
class SessionState:
    """
    Persisted session state.
    """
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Metrics
    tool_calls: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, int] = field(default_factory=dict)
    
    # Identity
    identity_version: str = "v0.0.0"
    introspection_count: int = 0
    
    # Plugin state
    plugin_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(**data)


class SessionManager:
    """
    Manages session state persistence.
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or Path.home() / ".eidosian" / "sessions"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state: Optional[SessionState] = None
        self._auto_save_interval = 60  # seconds
        self._auto_save_thread: Optional[threading.Thread] = None
        self._running = False
    
    @property
    def state_file(self) -> Path:
        return self.state_dir / "current_session.json"
    
    @property
    def history_dir(self) -> Path:
        hist = self.state_dir / "history"
        hist.mkdir(exist_ok=True)
        return hist
    
    def load_or_create(self, session_id: str) -> SessionState:
        """Load existing state or create new."""
        with self._lock:
            if self.state_file.exists():
                try:
                    data = json.loads(self.state_file.read_text())
                    self._state = SessionState.from_dict(data)
                    self._state.last_updated = datetime.now(timezone.utc).isoformat()
                except Exception:
                    self._state = SessionState(session_id=session_id)
            else:
                self._state = SessionState(session_id=session_id)
            
            return self._state
    
    def save(self) -> None:
        """Save current state."""
        with self._lock:
            if self._state:
                self._state.last_updated = datetime.now(timezone.utc).isoformat()
                self.state_file.write_text(json.dumps(self._state.to_dict(), indent=2))
    
    def archive(self) -> Path:
        """Archive current session and start fresh."""
        with self._lock:
            if self._state:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                archive_path = self.history_dir / f"session_{timestamp}.json"
                archive_path.write_text(json.dumps(self._state.to_dict(), indent=2))
                self._state = None
                if self.state_file.exists():
                    self.state_file.unlink()
                return archive_path
        return self.history_dir / "no_session.json"
    
    def record_tool_call(self, tool_name: str, success: bool = True) -> None:
        """Record a tool call."""
        with self._lock:
            if self._state:
                self._state.tool_calls[tool_name] = self._state.tool_calls.get(tool_name, 0) + 1
                if not success:
                    self._state.errors[tool_name] = self._state.errors.get(tool_name, 0) + 1
    
    def update_identity(self, version: str, introspection_count: int) -> None:
        """Update identity metrics."""
        with self._lock:
            if self._state:
                self._state.identity_version = version
                self._state.introspection_count = introspection_count
    
    def update_plugin_metrics(self, plugin_id: str, metrics: Dict[str, Any]) -> None:
        """Update plugin-specific metrics."""
        with self._lock:
            if self._state:
                self._state.plugin_metrics[plugin_id] = metrics
    
    def set_custom(self, key: str, value: Any) -> None:
        """Set custom data."""
        with self._lock:
            if self._state:
                self._state.custom_data[key] = value
    
    def get_custom(self, key: str, default: Any = None) -> Any:
        """Get custom data."""
        with self._lock:
            if self._state:
                return self._state.custom_data.get(key, default)
            return default
    
    def start_auto_save(self) -> None:
        """Start auto-save background thread."""
        if self._running:
            return
        
        self._running = True
        
        def auto_save_loop():
            while self._running:
                time.sleep(self._auto_save_interval)
                if self._running:
                    self.save()
        
        self._auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        self._auto_save_thread.start()
    
    def stop_auto_save(self) -> None:
        """Stop auto-save background thread."""
        self._running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        with self._lock:
            if self._state:
                return {
                    "session_id": self._state.session_id,
                    "started_at": self._state.started_at,
                    "last_updated": self._state.last_updated,
                    "total_tool_calls": sum(self._state.tool_calls.values()),
                    "total_errors": sum(self._state.errors.values()),
                    "identity_version": self._state.identity_version,
                    "introspection_count": self._state.introspection_count,
                    "plugins_tracked": len(self._state.plugin_metrics),
                    "auto_save_running": self._running
                }
            return {"error": "No active session"}
    
    def list_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List historical sessions."""
        sessions = []
        for f in sorted(self.history_dir.glob("session_*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(f.read_text())
                sessions.append({
                    "file": f.name,
                    "session_id": data.get("session_id"),
                    "started_at": data.get("started_at"),
                    "total_tool_calls": sum(data.get("tool_calls", {}).values())
                })
            except Exception:
                pass
        return sessions


# Global session manager
_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    return _session_manager


def session_status() -> str:
    """Get current session status."""
    return json.dumps({
        "status": "success",
        **_session_manager.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, indent=2)


def session_history(limit: int = 10) -> str:
    """Get historical sessions."""
    return json.dumps({
        "status": "success",
        "sessions": _session_manager.list_history(limit),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }, indent=2)


def session_archive() -> str:
    """Archive current session."""
    path = _session_manager.archive()
    return json.dumps({
        "status": "success",
        "archived_to": str(path),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


if __name__ == "__main__":
    import uuid
    
    print("🧪 Testing Session Manager\n")
    
    # Create session
    session_id = str(uuid.uuid4())[:8]
    state = _session_manager.load_or_create(session_id)
    print(f"Session created: {session_id}")
    
    # Record some activity
    _session_manager.record_tool_call("memory_add")
    _session_manager.record_tool_call("memory_add")
    _session_manager.record_tool_call("kb_search")
    _session_manager.record_tool_call("web_fetch", success=False)
    
    # Update identity
    _session_manager.update_identity("v0.5.0", 24)
    
    # Save
    _session_manager.save()
    
    # Print status
    print("\nSession status:")
    print(session_status())

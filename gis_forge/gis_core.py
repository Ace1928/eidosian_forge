"""
GIS Forge - Global Information System
Centralized configuration registry with hierarchical inheritance.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

class GisCore:
    """
    Core implementation of the Eidosian Global Information System.
    Provides a thread-safe, hierarchical configuration registry with persistence.
    """
    
    def __init__(self, persistence_path: Optional[Union[str, Path]] = None):
        self._registry: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable[[str, Any], None]]] = {}
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        if self.persistence_path and self.persistence_path.exists():
            self.load_json(self.persistence_path)
            
    def set(self, key: str, value: Any, notify: bool = True, persist: bool = True) -> None:
        """
        Set a value in the registry. 
        Supports dot-notation for nested structures (e.g., 'database.host').
        """
        with self._lock:
            parts = key.split('.')
            target = self._registry
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            
            target[parts[-1]] = value
            
        if notify:
            self._notify(key, value)
            
        if persist and self.persistence_path:
            self.save_json(self.persistence_path)

    def get(self, key: str, default: Any = None, use_env: bool = True) -> Any:
        """
        Get a value from the registry using dot-notation.
        If use_env is True, checks for EIDOS_ prefixed environment variables first.
        Example: get('db.host') checks EIDOS_DB_HOST
        """
        if use_env:
            env_key = f"EIDOS_{key.upper().replace('.', '_')}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    # Try to parse as JSON for complex types, fallback to string
                    return json.loads(env_val)
                except json.JSONDecodeError:
                    return env_val

        with self._lock:
            parts = key.split('.')
            target = self._registry
            for part in parts:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return default
            return target

    def delete(self, key: str, persist: bool = True) -> bool:
        """Remove a key from the registry."""
        with self._lock:
            parts = key.split('.')
            target = self._registry
            for part in parts[:-1]:
                if isinstance(target, dict) and part in target:
                    target = target[part]
                else:
                    return False
            
            if isinstance(target, dict) and parts[-1] in target:
                del target[parts[-1]]
                if persist and self.persistence_path:
                    self.save_json(self.persistence_path)
                return True
        return False

    def update(self, config: Dict[str, Any], prefix: str = "", persist: bool = True) -> None:
        """
        Bulk update the registry from a dictionary.
        """
        for k, v in config.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                self.update(v, full_key, persist=False)
            else:
                self.set(full_key, v, persist=False)
        
        if persist and self.persistence_path:
            self.save_json(self.persistence_path)

    def subscribe(self, key_prefix: str, callback: Callable[[str, Any], None]) -> None:
        """
        Subscribe to changes on keys starting with key_prefix.
        """
        with self._lock:
            if key_prefix not in self._subscribers:
                self._subscribers[key_prefix] = []
            self._subscribers[key_prefix].append(callback)

    def _notify(self, key: str, value: Any) -> None:
        """Notify subscribers of a change."""
        callbacks_to_run = []
        with self._lock:
            for prefix, callbacks in self._subscribers.items():
                if key.startswith(prefix):
                    callbacks_to_run.extend(callbacks)
        
        for callback in callbacks_to_run:
            try:
                callback(key, value)
            except Exception as e:
                logging.error(f"Error in GIS subscriber for {key}: {e}")

    def load_json(self, path: Union[str, Path], prefix: str = "") -> None:
        """Load configuration from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            try:
                data = json.load(f)
                self.update(data, prefix, persist=False)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode GIS persistence file {path}: {e}")

    def save_json(self, path: Union[str, Path]) -> None:
        """Save the current registry to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(path, 'w') as f:
                json.dump(self._registry, f, indent=2)

    def flatten(self) -> Dict[str, Any]:
        """Return a flattened version of the registry (dot-notation keys)."""
        result = {}
        def _walk(d, prefix=""):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _walk(v, full_key)
                else:
                    result[full_key] = v
        with self._lock:
            _walk(self._registry)
        return result

    def __repr__(self) -> str:
        with self._lock:
            return f"<GisCore keys={list(self._registry.keys())}>"

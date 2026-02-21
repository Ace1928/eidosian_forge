from __future__ import annotations

from typing import Any

from type_forge import TypeCore


class _FakeBackend:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str, default: Any = None, use_env: bool = True) -> Any:
        return self.store.get(key, default)

    def set(self, key: str, value: Any, notify: bool = True, persist: bool = True) -> None:
        self.store[key] = value


def test_registry_persisted_after_register_delete() -> None:
    backend = _FakeBackend()
    core = TypeCore(registry_backend=backend, registry_key="types.schemas")
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    changed = core.register_schema("person", schema)
    assert changed is True
    assert backend.store["types.schemas"]["person"] == schema

    removed = core.delete_schema("person")
    assert removed is True
    assert backend.store["types.schemas"] == {}


def test_registry_restored_from_backend_on_startup() -> None:
    backend = _FakeBackend()
    schema = {"type": "object", "properties": {"v": {"type": "integer"}}, "required": ["v"]}
    backend.store["types.schemas"] = {"counter": schema}

    core = TypeCore(registry_backend=backend, registry_key="types.schemas")

    assert core.get_schema("counter") == schema

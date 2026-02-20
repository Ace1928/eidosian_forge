from __future__ import annotations

import json
from pathlib import Path

from eidosian_core import (
    clear_port_registry_cache,
    get_service_host,
    get_service_port,
    get_service_url,
    load_port_registry,
)


def _write_registry(path: Path) -> None:
    payload = {
        "services": {
            "svc": {
                "protocol": "http",
                "host": "127.0.0.7",
                "port": 17777,
                "path": "/api",
                "env": ["EIDOS_SVC_PORT"],
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_get_service_port_prefers_env(monkeypatch, tmp_path: Path) -> None:
    registry = tmp_path / "ports.json"
    _write_registry(registry)

    clear_port_registry_cache()
    monkeypatch.setenv("EIDOS_SVC_PORT", "19999")
    assert get_service_port("svc", default=1000, registry_path=str(registry)) == 19999


def test_get_service_port_uses_registry_when_env_unset(monkeypatch, tmp_path: Path) -> None:
    registry = tmp_path / "ports.json"
    _write_registry(registry)

    clear_port_registry_cache()
    monkeypatch.delenv("EIDOS_SVC_PORT", raising=False)
    assert get_service_port("svc", default=1000, registry_path=str(registry)) == 17777


def test_get_service_url_builds_from_registry(tmp_path: Path) -> None:
    registry = tmp_path / "ports.json"
    _write_registry(registry)

    clear_port_registry_cache()
    assert get_service_host("svc", registry_path=str(registry)) == "127.0.0.7"
    assert get_service_url("svc", default_port=9000, registry_path=str(registry)) == "http://127.0.0.7:17777/api"


def test_load_port_registry_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    clear_port_registry_cache()
    loaded = load_port_registry(str(missing))
    assert loaded == {"services": {}}

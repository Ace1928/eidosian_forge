from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REGISTRY_RELATIVE_PATH = Path("config/ports.json")


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("EIDOS_FORGE_ROOT", "EIDOS_FORGE_DIR"):
        raw = os.environ.get(key)
        if raw:
            roots.append(Path(raw).expanduser().resolve())

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        roots.append(parent)

    roots.append(Path.cwd().resolve())

    deduped: list[Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item)
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def detect_forge_root() -> Path:
    for root in _candidate_roots():
        if (root / DEFAULT_REGISTRY_RELATIVE_PATH).exists():
            return root
    env_root = os.environ.get("EIDOS_FORGE_ROOT") or os.environ.get("EIDOS_FORGE_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def detect_registry_path() -> Path:
    override = os.environ.get("EIDOS_PORT_REGISTRY_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return detect_forge_root() / DEFAULT_REGISTRY_RELATIVE_PATH


@lru_cache(maxsize=4)
def load_port_registry(path: str | None = None) -> dict[str, Any]:
    registry_path = Path(path).expanduser().resolve() if path else detect_registry_path()
    if not registry_path.exists():
        return {"services": {}}
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {"services": {}}
    if not isinstance(payload, dict):
        return {"services": {}}
    services = payload.get("services")
    if not isinstance(services, dict):
        payload["services"] = {}
    return payload


def clear_port_registry_cache() -> None:
    load_port_registry.cache_clear()


def get_service_config(service_key: str, registry_path: str | None = None) -> dict[str, Any]:
    registry = load_port_registry(registry_path)
    services = registry.get("services")
    if not isinstance(services, dict):
        return {}
    service = services.get(service_key)
    return service if isinstance(service, dict) else {}


def _first_non_empty_env(keys: Iterable[str]) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


def get_service_port(
    service_key: str,
    *,
    default: int,
    env_keys: Iterable[str] | None = None,
    registry_path: str | None = None,
) -> int:
    service = get_service_config(service_key, registry_path)
    candidates: list[str] = []

    if env_keys:
        candidates.extend(list(env_keys))

    service_env = service.get("env")
    if isinstance(service_env, list):
        candidates.extend([str(item) for item in service_env if isinstance(item, str)])

    env_raw = _first_non_empty_env(candidates)
    env_port = _parse_int(env_raw)
    if env_port is not None:
        return env_port

    service_port = _parse_int(str(service.get("port")))
    if service_port is not None:
        return service_port

    return int(default)


def get_service_host(
    service_key: str,
    *,
    default: str = "127.0.0.1",
    env_keys: Iterable[str] | None = None,
    registry_path: str | None = None,
) -> str:
    service = get_service_config(service_key, registry_path)
    candidates: list[str] = []

    if env_keys:
        candidates.extend(list(env_keys))

    env_raw = _first_non_empty_env(candidates)
    if env_raw:
        return env_raw

    host = service.get("host")
    if isinstance(host, str) and host.strip():
        return host.strip()

    return default


def get_service_path(service_key: str, *, default: str = "", registry_path: str | None = None) -> str:
    service = get_service_config(service_key, registry_path)
    raw = service.get("path")
    if isinstance(raw, str):
        return raw
    return default


def get_service_url(
    service_key: str,
    *,
    default_port: int,
    default_host: str = "127.0.0.1",
    default_protocol: str = "http",
    default_path: str = "",
    registry_path: str | None = None,
) -> str:
    service = get_service_config(service_key, registry_path)
    protocol = service.get("protocol") if isinstance(service.get("protocol"), str) else default_protocol
    host = get_service_host(service_key, default=default_host, registry_path=registry_path)
    port = get_service_port(service_key, default=default_port, registry_path=registry_path)
    path = service.get("path") if isinstance(service.get("path"), str) else default_path
    return f"{protocol}://{host}:{port}{path}"


def should_use_registry_fallback() -> bool:
    """Compatibility helper: whether env fallback behavior should be enabled.

    Always True by default, but can be disabled by setting EIDOS_PORT_REGISTRY_DISABLE=1.
    """
    return not _truthy(os.environ.get("EIDOS_PORT_REGISTRY_DISABLE"))

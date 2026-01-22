from __future__ import annotations
import logging
from collections.abc import Callable
from qiskit.providers.backend import Backend
def resolve_backend_name(name: str, backends: list[Backend], deprecated: dict[str, str], aliased: dict[str, list[str]]) -> str:
    """Resolve backend name from a deprecated name or an alias.

    A group will be resolved in order of member priorities, depending on
    availability.

    Args:
        name (str): name of backend to resolve
        backends (list[Backend]): list of available backends.
        deprecated (dict[str: str]): dict of deprecated names.
        aliased (dict[str: list[str]]): dict of aliased names.

    Returns:
        str: resolved name (name of an available backend)

    Raises:
        LookupError: if name cannot be resolved through regular available
            names, nor deprecated, nor alias names.
    """
    available = []
    for backend in backends:
        available.append(backend.name() if backend.version == 1 else backend.name)
    resolved_name = deprecated.get(name, aliased.get(name, name))
    if isinstance(resolved_name, list):
        resolved_name = next((b for b in resolved_name if b in available), '')
    if resolved_name not in available:
        raise LookupError(f"backend '{name}' not found.")
    if name in deprecated:
        logger.warning("Backend '%s' is deprecated. Use '%s'.", name, resolved_name)
    return resolved_name
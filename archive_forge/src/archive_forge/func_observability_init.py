from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
def observability_init(observability_plugin: ObservabilityPlugin) -> None:
    """Initialize observability with provided ObservabilityPlugin.

    This method have to be called at the start of a program, before any
    channels/servers are built.

    Args:
    observability_plugin: The ObservabilityPlugin to use.

    Raises:
      ValueError: If an ObservabilityPlugin was already registered at the
      time of calling this method.
    """
    set_plugin(observability_plugin)
    try:
        _cygrpc.set_server_call_tracer_factory(observability_plugin)
    except Exception:
        _LOGGER.exception('Failed to set server call tracer factory!')
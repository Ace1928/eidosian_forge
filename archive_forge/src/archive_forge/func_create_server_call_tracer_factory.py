from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
@abc.abstractmethod
def create_server_call_tracer_factory(self) -> ServerCallTracerFactoryCapsule:
    """Creates a ServerCallTracerFactoryCapsule.

        After register the plugin, if tracing or stats is enabled, this method
        will be called by calling observability_init, the ServerCallTracerFactory
        created by this method will be registered to gRPC core.

        The ServerCallTracerFactory is an object which implements
        `grpc_core::ServerCallTracerFactory` interface and wrapped in a PyCapsule
        using `server_call_tracer_factory` as name.

        Returns:
        A PyCapsule which stores a ServerCallTracerFactory object.
        """
    raise NotImplementedError()
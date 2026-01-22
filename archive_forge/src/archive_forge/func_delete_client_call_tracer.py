from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
@abc.abstractmethod
def delete_client_call_tracer(self, client_call_tracer: ClientCallTracerCapsule) -> None:
    """Deletes the ClientCallTracer stored in ClientCallTracerCapsule.

        After register the plugin, if tracing or stats is enabled, this method
        will be called at the end of the call to destroy the ClientCallTracer.

        The ClientCallTracer is an object which implements `grpc_core::ClientCallTracer`
        interface and wrapped in a PyCapsule using `client_call_tracer` as name.

        Args:
        client_call_tracer: A PyCapsule which stores a ClientCallTracer object.
        """
    raise NotImplementedError()
from __future__ import annotations
import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar
from grpc._cython import cygrpc as _cygrpc
@abc.abstractmethod
def record_rpc_latency(self, method: str, target: str, rpc_latency: float, status_code: Any) -> None:
    """Record the latency of the RPC.

        After register the plugin, if stats is enabled, this method will be
        called at the end of each RPC.

        Args:
        method: The fully-qualified name of the RPC method being invoked.
        target: The target name of the RPC method being invoked.
        rpc_latency: The latency for the RPC in seconds, equals to the time between
         when the client invokes the RPC and when the client receives the status.
        status_code: An element of grpc.StatusCode in string format representing the
         final status for the RPC.
        """
    raise NotImplementedError()
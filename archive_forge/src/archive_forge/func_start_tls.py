import enum
import logging
import ssl
import time
from types import TracebackType
from typing import (
import h11
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def start_tls(self, ssl_context: ssl.SSLContext, server_hostname: Optional[str]=None, timeout: Optional[float]=None) -> NetworkStream:
    return self._stream.start_tls(ssl_context, server_hostname, timeout)
import itertools
import logging
import ssl
from types import TracebackType
from typing import Iterable, Iterator, Optional, Type
from .._backends.auto import AutoBackend
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream
from .._exceptions import ConnectError, ConnectTimeout
from .._models import Origin, Request, Response
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface
def exponential_backoff(factor: float) -> Iterator[float]:
    """
    Generate a geometric sequence that has a ratio of 2 and starts with 0.

    For example:
    - `factor = 2`: `0, 2, 4, 8, 16, 32, 64, ...`
    - `factor = 3`: `0, 3, 6, 12, 24, 48, 96, ...`
    """
    yield 0
    for n in itertools.count():
        yield (factor * 2 ** n)
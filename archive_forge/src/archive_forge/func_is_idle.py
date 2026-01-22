import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import AsyncNetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import AsyncLock, AsyncSemaphore, AsyncShieldCancellation
from .._trace import Trace
from .interfaces import AsyncConnectionInterface
def is_idle(self) -> bool:
    return self._state == HTTPConnectionState.IDLE
from __future__ import annotations
import time
import typing
from enum import Enum
from socket import getdefaulttimeout
from ..exceptions import TimeoutStateError
@staticmethod
def resolve_default_timeout(timeout: _TYPE_TIMEOUT) -> float | None:
    return getdefaulttimeout() if timeout is _DEFAULT_TIMEOUT else timeout
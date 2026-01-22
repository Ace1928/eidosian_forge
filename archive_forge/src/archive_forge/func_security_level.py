import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
@property
def security_level(self) -> int:
    return self._ctx.security_level
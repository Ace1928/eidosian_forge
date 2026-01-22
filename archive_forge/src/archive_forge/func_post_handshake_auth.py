import os
import platform
import socket
import ssl
import typing
import _ssl  # type: ignore[import]
from ._ssl_constants import (
@post_handshake_auth.setter
def post_handshake_auth(self, value: bool) -> None:
    self._ctx.post_handshake_auth = value
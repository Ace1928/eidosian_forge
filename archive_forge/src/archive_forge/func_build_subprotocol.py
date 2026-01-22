from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def build_subprotocol(subprotocols: Sequence[Subprotocol]) -> str:
    """
    Build a ``Sec-WebSocket-Protocol`` header.

    This is the reverse of :func:`parse_subprotocol`.

    """
    return ', '.join(subprotocols)
from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_subprotocol(header: str) -> List[Subprotocol]:
    """
    Parse a ``Sec-WebSocket-Protocol`` header.

    Return a list of WebSocket subprotocols.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    return parse_list(parse_subprotocol_item, header, 0, 'Sec-WebSocket-Protocol')
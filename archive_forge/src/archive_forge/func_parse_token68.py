from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def parse_token68(header: str, pos: int, header_name: str) -> Tuple[str, int]:
    """
    Parse a token68 from ``header`` at the given position.

    Return the token value and the new position.

    Raises:
        InvalidHeaderFormat: on invalid inputs.

    """
    match = _token68_re.match(header, pos)
    if match is None:
        raise exceptions.InvalidHeaderFormat(header_name, 'expected token68', header, pos)
    return (match.group(), match.end())
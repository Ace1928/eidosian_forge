from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def peek_ahead(header: str, pos: int) -> Optional[str]:
    """
    Return the next character from ``header`` at the given position.

    Return :obj:`None` at the end of ``header``.

    We never need to peek more than one character ahead.

    """
    return None if pos == len(header) else header[pos]
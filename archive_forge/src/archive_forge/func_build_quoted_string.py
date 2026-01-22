from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def build_quoted_string(value: str) -> str:
    """
    Format ``value`` as a quoted string.

    This is the reverse of :func:`parse_quoted_string`.

    """
    match = _quotable_re.fullmatch(value)
    if match is None:
        raise ValueError('invalid characters for quoted-string encoding')
    return '"' + _quote_re.sub('\\\\\\1', value) + '"'
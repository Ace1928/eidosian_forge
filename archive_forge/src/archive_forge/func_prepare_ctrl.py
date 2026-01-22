from __future__ import annotations
import dataclasses
import enum
import io
import secrets
import struct
from typing import Callable, Generator, Optional, Sequence, Tuple
from . import exceptions, extensions
from .typing import Data
def prepare_ctrl(data: Data) -> bytes:
    """
    Convert a string or byte-like object to bytes.

    This function is designed for ping and pong frames.

    If ``data`` is a :class:`str`, return a :class:`bytes` object encoding
    ``data`` in UTF-8.

    If ``data`` is a bytes-like object, return a :class:`bytes` object.

    Raises:
        TypeError: if ``data`` doesn't have a supported type.

    """
    if isinstance(data, str):
        return data.encode('utf-8')
    elif isinstance(data, BytesLike):
        return bytes(data)
    else:
        raise TypeError('data must be str or bytes-like')
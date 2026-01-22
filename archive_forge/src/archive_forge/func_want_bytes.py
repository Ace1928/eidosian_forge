from __future__ import annotations
import base64
import string
import struct
import typing as t
from .exc import BadData
def want_bytes(s: str | bytes, encoding: str='utf-8', errors: str='strict') -> bytes:
    if isinstance(s, str):
        s = s.encode(encoding, errors)
    return s
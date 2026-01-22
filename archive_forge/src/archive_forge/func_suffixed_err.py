from __future__ import annotations
from collections.abc import Iterable
import string
from types import MappingProxyType
from typing import Any, BinaryIO, NamedTuple
from ._re import (
from ._types import Key, ParseFloat, Pos
def suffixed_err(src: str, pos: Pos, msg: str) -> TOMLDecodeError:
    """Return a `TOMLDecodeError` where error message is suffixed with
    coordinates in source."""

    def coord_repr(src: str, pos: Pos) -> str:
        if pos >= len(src):
            return 'end of document'
        line = src.count('\n', 0, pos) + 1
        if line == 1:
            column = pos + 1
        else:
            column = pos - src.rindex('\n', 0, pos)
        return f'line {line}, column {column}'
    return TOMLDecodeError(f'{msg} (at {coord_repr(src, pos)})')
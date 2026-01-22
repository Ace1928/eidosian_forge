from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def charCodeAt(src: str, pos: int) -> int | None:
    """
    Returns the Unicode value of the character at the specified location.

    @param - index The zero-based index of the desired character.
    If there is no character at the specified index, NaN is returned.

    This was added for compatibility with python
    """
    try:
        return ord(src[pos])
    except IndexError:
        return None
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def skipChars(self, pos: int, code: int) -> int:
    """Skip character code from given position."""
    while True:
        try:
            current = self.srcCharCode[pos]
        except IndexError:
            break
        if current != code:
            break
        pos += 1
    return pos
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
def skipEmptyLines(self, from_pos: int) -> int:
    """."""
    while from_pos < self.lineMax:
        try:
            if self.bMarks[from_pos] + self.tShift[from_pos] < self.eMarks[from_pos]:
                break
        except IndexError:
            pass
        from_pos += 1
    return from_pos
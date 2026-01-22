from __future__ import annotations
from ...interpreterbase import (
import typing as T
@typed_operator(MesonOperator.MOD, int)
def op_mod(self, other: int) -> int:
    if other == 0:
        raise InvalidArguments('Tried to divide by 0')
    return self.held_object % other
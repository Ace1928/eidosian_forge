from __future__ import annotations
from ...interpreterbase import (
import typing as T
@noKwargs
@noPosargs
def to_int_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> int:
    return 1 if self.held_object else 0
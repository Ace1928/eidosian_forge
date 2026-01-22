from __future__ import annotations
import typing as T
from ...interpreterbase import (
from ...mparser import PlusAssignmentNode
@noKwargs
@noPosargs
def length_method(self, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> int:
    return len(self.held_object)
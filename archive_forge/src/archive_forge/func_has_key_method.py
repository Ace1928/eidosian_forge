from __future__ import annotations
import typing as T
from ...interpreterbase import (
@noKwargs
@typed_pos_args('dict.has_key', str)
def has_key_method(self, args: T.Tuple[str], kwargs: TYPE_kwargs) -> bool:
    return args[0] in self.held_object
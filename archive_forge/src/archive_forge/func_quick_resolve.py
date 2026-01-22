from __future__ import annotations
import os
import sys
import typing as T
from .. import mparser, mesonlib
from .. import environment
from ..interpreterbase import (
from ..interpreter import (
from ..mparser import (
def quick_resolve(n: BaseNode, loop_detect: T.Optional[T.List[str]]=None) -> T.Any:
    if loop_detect is None:
        loop_detect = []
    if isinstance(n, IdNode):
        assert isinstance(n.value, str)
        if n.value in loop_detect or n.value not in self.assignments:
            return []
        return quick_resolve(self.assignments[n.value], loop_detect=loop_detect + [n.value])
    elif isinstance(n, ElementaryNode):
        return n.value
    else:
        return n
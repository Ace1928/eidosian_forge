from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
def visit_dml_values(self, attrname, left_parent, left, right_parent, right, **kw):
    if left is None or right is None or len(left) != len(right):
        return COMPARE_FAILED
    if isinstance(left, collections_abc.Sequence):
        for lv, rv in zip(left, right):
            if not self._compare_dml_values_or_ce(lv, rv, **kw):
                return COMPARE_FAILED
    elif isinstance(right, collections_abc.Sequence):
        return COMPARE_FAILED
    else:
        for (lk, lv), (rk, rv) in zip(left.items(), right.items()):
            if not self._compare_dml_values_or_ce(lk, rk, **kw):
                return COMPARE_FAILED
            if not self._compare_dml_values_or_ce(lv, rv, **kw):
                return COMPARE_FAILED
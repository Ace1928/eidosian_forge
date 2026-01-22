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
def visit_table_hint_list(self, attrname, left_parent, left, right_parent, right, **kw):
    left_keys = sorted(left, key=lambda elem: (elem[0].fullname, elem[1]))
    right_keys = sorted(right, key=lambda elem: (elem[0].fullname, elem[1]))
    for (ltable, ldialect), (rtable, rdialect) in zip_longest(left_keys, right_keys, fillvalue=(None, None)):
        if ldialect != rdialect:
            return COMPARE_FAILED
        elif left[ltable, ldialect] != right[rtable, rdialect]:
            return COMPARE_FAILED
        else:
            self.stack.append((ltable, rtable))
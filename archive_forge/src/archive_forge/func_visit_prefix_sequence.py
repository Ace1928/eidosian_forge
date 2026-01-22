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
def visit_prefix_sequence(self, attrname, left_parent, left, right_parent, right, **kw):
    for (l_clause, l_str), (r_clause, r_str) in zip_longest(left, right, fillvalue=(None, None)):
        if l_str != r_str:
            return COMPARE_FAILED
        else:
            self.stack.append((l_clause, r_clause))
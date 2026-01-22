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
def visit_clauseelement_tuples(self, attrname, left_parent, left, right_parent, right, **kw):
    for ltup, rtup in zip_longest(left, right, fillvalue=None):
        if ltup is None or rtup is None:
            return COMPARE_FAILED
        for l, r in zip_longest(ltup, rtup, fillvalue=None):
            self.stack.append((l, r))
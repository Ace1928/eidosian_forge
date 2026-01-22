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
def visit_executable_options(self, attrname, left_parent, left, right_parent, right, **kw):
    for l, r in zip_longest(left, right, fillvalue=None):
        if l is None:
            if r is not None:
                return COMPARE_FAILED
            else:
                continue
        elif r is None:
            return COMPARE_FAILED
        if (l._gen_cache_key(self.anon_map[0], []) if l._is_has_cache_key else l) != (r._gen_cache_key(self.anon_map[1], []) if r._is_has_cache_key else r):
            return COMPARE_FAILED
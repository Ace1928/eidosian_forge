import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
def wildcard_keys(self, sort=SortComponents.UNSORTED):
    _iter = _IndexedComponent_slice_iter(self, sort=sort)
    return (_iter.get_last_index_wildcards() for _ in _iter)
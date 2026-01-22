import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
def get_last_index_wildcards(self):
    """Get a tuple of the values in the wildcard positions for the most
        recent indices corresponding to the last component returned by
        each _slice_generator in the iter stack.

        """
    ans = sum((tuple((x.last_index[i] for i in range(len(x.last_index)) if i not in x.fixed)) for x in self._iter_stack if x is not None), ())
    if not ans:
        return UnindexedComponent_index
    if len(ans) == 1:
        return ans[0]
    else:
        return ans
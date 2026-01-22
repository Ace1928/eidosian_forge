import collections
from numba.core import types
@wrap
def merge_pop(ms):
    """
        Pop the top run from the merge stack.
        """
    return MergeState(ms.min_gallop, ms.keys, ms.values, ms.pending, ms.n - 1)
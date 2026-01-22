import collections
from numba.core import types
@wrap
def merge_collapse(ms, keys, values):
    """
        Examine the stack of runs waiting to be merged, merging adjacent runs
        until the stack invariants are re-established:

        1. len[-3] > len[-2] + len[-1]
        2. len[-2] > len[-1]

        An updated MergeState is returned.

        See listsort.txt for more info.
        """
    while ms.n > 1:
        pending = ms.pending
        n = ms.n - 2
        if n > 0 and pending[n - 1].size <= pending[n].size + pending[n + 1].size or (n > 1 and pending[n - 2].size <= pending[n - 1].size + pending[n].size):
            if pending[n - 1].size < pending[n + 1].size:
                n -= 1
            ms = merge_at(ms, keys, values, n)
        elif pending[n].size < pending[n + 1].size:
            ms = merge_at(ms, keys, values, n)
        else:
            break
    return ms
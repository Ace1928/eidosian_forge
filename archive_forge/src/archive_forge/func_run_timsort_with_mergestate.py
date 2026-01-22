import collections
from numba.core import types
@wrap
def run_timsort_with_mergestate(ms, keys, values):
    """
        Run timsort with the mergestate.
        """
    nremaining = len(keys)
    if nremaining < 2:
        return
    minrun = merge_compute_minrun(nremaining)
    lo = zero
    while nremaining > 0:
        n, desc = count_run(keys, lo, lo + nremaining)
        if desc:
            reverse_slice(keys, values, lo, lo + n)
        if n < minrun:
            force = min(minrun, nremaining)
            binarysort(keys, values, lo, lo + force, lo + n)
            n = force
        ms = merge_append(ms, MergeRun(lo, n))
        ms = merge_collapse(ms, keys, values)
        lo += n
        nremaining -= n
    ms = merge_force_collapse(ms, keys, values)
    assert ms.n == 1
    assert ms.pending[0] == (0, len(keys))
import collections
from numba.core import types
@wrap
def sortslice_copy(dest_keys, dest_values, dest_start, src_keys, src_values, src_start, nitems):
    """
        Upwards memcpy().
        """
    assert src_start >= 0
    assert dest_start >= 0
    for i in range(nitems):
        dest_keys[dest_start + i] = src_keys[src_start + i]
    if has_values(src_keys, src_values):
        for i in range(nitems):
            dest_values[dest_start + i] = src_values[src_start + i]
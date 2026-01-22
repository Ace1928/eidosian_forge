import collections
from numba.core import types
@wrap
def reverse_slice(keys, values, start, stop):
    """
        Reverse a slice, in-place.
        """
    i = start
    j = stop - 1
    while i < j:
        keys[i], keys[j] = (keys[j], keys[i])
        i += 1
        j -= 1
    if has_values(keys, values):
        i = start
        j = stop - 1
        while i < j:
            values[i], values[j] = (values[j], values[i])
            i += 1
            j -= 1
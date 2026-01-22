import collections
from numba.core import types
@wrap
def merge_init_with_values(keys, values):
    """
        Initialize a MergeState for a keyed sort.
        """
    temp_size = min(len(keys) // 2 + 1, MERGESTATE_TEMP_SIZE)
    temp_keys = make_temp_area(keys, temp_size)
    temp_values = make_temp_area(values, temp_size)
    pending = [MergeRun(zero, zero)] * MAX_MERGE_PENDING
    return MergeState(intp(MIN_GALLOP), temp_keys, temp_values, pending, zero)
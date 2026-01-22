import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
@trace.trace_wrapper
def assert_no_leak_if_all_possibly_except_one(self):
    """Raises an exception if a leak is detected.

    This algorithm classifies a series of allocations as a leak if it's the same
    type at every snapshot, but possibly except one snapshot.
    """
    snapshot_diffs = []
    for i in range(0, len(self._snapshots) - 1):
        snapshot_diffs.append(self._snapshot_diff(i, i + 1))
    allocation_counter = collections.Counter()
    for diff in snapshot_diffs:
        for name, count in diff.items():
            if count > 0:
                allocation_counter[name] += 1
    leaking_object_names = {name for name, count in allocation_counter.items() if count >= len(snapshot_diffs) - 1}
    if leaking_object_names:
        object_list_to_print = '\n'.join([' - ' + name for name in leaking_object_names])
        raise AssertionError(f'These Python objects were allocated in every snapshot possibly except one.\n\n{object_list_to_print}')
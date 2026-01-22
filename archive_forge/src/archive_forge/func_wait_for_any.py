import collections
import contextlib
import functools
from concurrent import futures
from concurrent.futures import _base
import futurist
from futurist import _utils
def wait_for_any(fs, timeout=None):
    """Wait for one (**any**) of the futures to complete.

    Works correctly with both green and non-green futures (but not both
    together, since this can't be guaranteed to avoid dead-lock due to how
    the waiting implementations are different when green threads are being
    used).

    Returns pair (done futures, not done futures).
    """
    return _wait_for(fs, futures.FIRST_COMPLETED, _wait_for_any_green, 'wait_for_any', timeout=timeout)
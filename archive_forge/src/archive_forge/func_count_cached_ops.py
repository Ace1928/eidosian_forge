import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
def count_cached_ops(cache):
    """Returns a counter of the types of each op in the cache.
    This is useful for profiling to increase sharing.
    """
    return Counter((key[0] for key in cache.keys()))
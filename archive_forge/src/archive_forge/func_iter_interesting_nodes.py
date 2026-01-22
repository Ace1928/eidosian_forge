import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def iter_interesting_nodes(store, interesting_root_keys, uninteresting_root_keys, pb=None):
    """Given root keys, find interesting nodes.

    Evaluate nodes referenced by interesting_root_keys. Ones that are also
    referenced from uninteresting_root_keys are not considered interesting.

    :param interesting_root_keys: keys which should be part of the
        "interesting" nodes (which will be yielded)
    :param uninteresting_root_keys: keys which should be filtered out of the
        result set.
    :return: Yield
        (interesting record, {interesting key:values})
    """
    iterator = CHKMapDifference(store, interesting_root_keys, uninteresting_root_keys, search_key_func=store._search_key_func, pb=pb)
    return iterator.process()
import heapq
import threading
from typing import Callable
from .. import errors, lru_cache, osutils, registry, trace
from .static_tuple import StaticTuple, expect_static_tuple
def process_common_internal_nodes(self_node, basis_node):
    self_items = set(self_node._items.items())
    basis_items = set(basis_node._items.items())
    path = (self_node._key, None)
    for prefix, child in self_items - basis_items:
        heapq.heappush(self_pending, (prefix, None, child, path))
    path = (basis_node._key, None)
    for prefix, child in basis_items - self_items:
        heapq.heappush(basis_pending, (prefix, None, child, path))
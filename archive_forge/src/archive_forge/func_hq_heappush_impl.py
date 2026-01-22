import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def hq_heappush_impl(heap, item):
    heap.append(item)
    _siftdown(heap, 0, len(heap) - 1)
import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
@overload(hq.heapify)
def hq_heapify(x):
    assert_heap_type(x)

    def hq_heapify_impl(x):
        n = len(x)
        for i in reversed_range(n // 2):
            _siftup(x, i)
    return hq_heapify_impl
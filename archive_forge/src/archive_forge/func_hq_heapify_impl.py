import heapq as hq
from numba.core import types
from numba.core.errors import TypingError
from numba.core.extending import overload, register_jitable
def hq_heapify_impl(x):
    n = len(x)
    for i in reversed_range(n // 2):
        _siftup(x, i)
import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
def zip_equal(*iterables):
    """``zip`` the input *iterables* together, but raise
    ``UnequalIterablesError`` if they aren't all the same length.

        >>> it_1 = range(3)
        >>> it_2 = iter('abc')
        >>> list(zip_equal(it_1, it_2))
        [(0, 'a'), (1, 'b'), (2, 'c')]

        >>> it_1 = range(3)
        >>> it_2 = iter('abcd')
        >>> list(zip_equal(it_1, it_2)) # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        more_itertools.more.UnequalIterablesError: Iterables have different
        lengths

    """
    if hexversion >= 50987174:
        warnings.warn('zip_equal will be removed in a future version of more-itertools. Use the builtin zip function with strict=True instead.', DeprecationWarning)
    return _zip_equal(*iterables)
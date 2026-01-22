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
def minmax(iterable_or_value, *others, key=None, default=_marker):
    """Returns both the smallest and largest items in an iterable
    or the largest of two or more arguments.

        >>> minmax([3, 1, 5])
        (1, 5)

        >>> minmax(4, 2, 6)
        (2, 6)

    If a *key* function is provided, it will be used to transform the input
    items for comparison.

        >>> minmax([5, 30], key=str)  # '30' sorts before '5'
        (30, 5)

    If a *default* value is provided, it will be returned if there are no
    input items.

        >>> minmax([], default=(0, 0))
        (0, 0)

    Otherwise ``ValueError`` is raised.

    This function is based on the
    `recipe <http://code.activestate.com/recipes/577916/>`__ by
    Raymond Hettinger and takes care to minimize the number of comparisons
    performed.
    """
    iterable = (iterable_or_value, *others) if others else iterable_or_value
    it = iter(iterable)
    try:
        lo = hi = next(it)
    except StopIteration as e:
        if default is _marker:
            raise ValueError('`minmax()` argument is an empty iterable. Provide a `default` value to suppress this error.') from e
        return default
    if key is None:
        for x, y in zip_longest(it, it, fillvalue=lo):
            if y < x:
                x, y = (y, x)
            if x < lo:
                lo = x
            if hi < y:
                hi = y
    else:
        lo_key = hi_key = key(lo)
        for x, y in zip_longest(it, it, fillvalue=lo):
            x_key, y_key = (key(x), key(y))
            if y_key < x_key:
                x, y, x_key, y_key = (y, x, y_key, x_key)
            if x_key < lo_key:
                lo, lo_key = (x, x_key)
            if hi_key < y_key:
                hi, hi_key = (y, y_key)
    return (lo, hi)
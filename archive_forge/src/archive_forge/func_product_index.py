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
def product_index(element, *args):
    """Equivalent to ``list(product(*args)).index(element)``

    The products of *args* can be ordered lexicographically.
    :func:`product_index` computes the first index of *element* without
    computing the previous products.

        >>> product_index([8, 2], range(10), range(5))
        42

    ``ValueError`` will be raised if the given *element* isn't in the product
    of *args*.
    """
    index = 0
    for x, pool in zip_longest(element, args, fillvalue=_marker):
        if x is _marker or pool is _marker:
            raise ValueError('element is not a product of args')
        pool = tuple(pool)
        index = index * len(pool) + pool.index(x)
    return index
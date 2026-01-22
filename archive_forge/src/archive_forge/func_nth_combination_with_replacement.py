import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import cached_property, partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log, perm, comb
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
def nth_combination_with_replacement(iterable, r, index):
    """Equivalent to
    ``list(combinations_with_replacement(iterable, r))[index]``.


    The subsequences with repetition of *iterable* that are of length *r* can
    be ordered lexicographically. :func:`nth_combination_with_replacement`
    computes the subsequence at sort position *index* directly, without
    computing the previous subsequences with replacement.

        >>> nth_combination_with_replacement(range(5), 3, 5)
        (0, 1, 1)

    ``ValueError`` will be raised If *r* is negative or greater than the length
    of *iterable*.
    ``IndexError`` will be raised if the given *index* is invalid.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r < 0 or r > n:
        raise ValueError
    c = comb(n + r - 1, r)
    if index < 0:
        index += c
    if index < 0 or index >= c:
        raise IndexError
    result = []
    i = 0
    while r:
        r -= 1
        while n >= 0:
            num_combs = comb(n + r - 1, r)
            if index < num_combs:
                break
            n -= 1
            i += 1
            index -= num_combs
        result.append(pool[i])
    return tuple(result)
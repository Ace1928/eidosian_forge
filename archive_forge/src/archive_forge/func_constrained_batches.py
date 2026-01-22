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
def constrained_batches(iterable, max_size, max_count=None, get_len=len, strict=True):
    """Yield batches of items from *iterable* with a combined size limited by
    *max_size*.

    >>> iterable = [b'12345', b'123', b'12345678', b'1', b'1', b'12', b'1']
    >>> list(constrained_batches(iterable, 10))
    [(b'12345', b'123'), (b'12345678', b'1', b'1'), (b'12', b'1')]

    If a *max_count* is supplied, the number of items per batch is also
    limited:

    >>> iterable = [b'12345', b'123', b'12345678', b'1', b'1', b'12', b'1']
    >>> list(constrained_batches(iterable, 10, max_count = 2))
    [(b'12345', b'123'), (b'12345678', b'1'), (b'1', b'12'), (b'1',)]

    If a *get_len* function is supplied, use that instead of :func:`len` to
    determine item size.

    If *strict* is ``True``, raise ``ValueError`` if any single item is bigger
    than *max_size*. Otherwise, allow single items to exceed *max_size*.
    """
    if max_size <= 0:
        raise ValueError('maximum size must be greater than zero')
    batch = []
    batch_size = 0
    batch_count = 0
    for item in iterable:
        item_len = get_len(item)
        if strict and item_len > max_size:
            raise ValueError('item size exceeds maximum size')
        reached_count = batch_count == max_count
        reached_size = item_len + batch_size > max_size
        if batch_count and (reached_size or reached_count):
            yield tuple(batch)
            batch.clear()
            batch_size = 0
            batch_count = 0
        batch.append(item)
        batch_size += item_len
        batch_count += 1
    if batch:
        yield tuple(batch)
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
def duplicates_justseen(iterable, key=None):
    """Yields serially-duplicate elements after their first appearance.

    >>> list(duplicates_justseen('mississippi'))
    ['s', 's', 'p']
    >>> list(duplicates_justseen('AaaBbbCccAaa', str.lower))
    ['a', 'a', 'b', 'b', 'c', 'c', 'a', 'a']

    This function is analagous to :func:`unique_justseen`.

    """
    return flatten(map(lambda group_tuple: islice_extended(group_tuple[1])[1:], groupby(iterable, key)))
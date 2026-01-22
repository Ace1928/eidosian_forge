import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def powerset(iterable):
    """Yields all possible subsets of the iterable.

        >>> list(powerset([1, 2, 3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    :func:`powerset` will operate on iterables that aren't :class:`set`
    instances, so repeated elements in the input will produce repeated elements
    in the output. Use :func:`unique_everseen` on the input to avoid generating
    duplicates:

        >>> seq = [1, 1, 0]
        >>> list(powerset(seq))
        [(), (1,), (1,), (0,), (1, 1), (1, 0), (1, 0), (1, 1, 0)]
        >>> from more_itertools import unique_everseen
        >>> list(powerset(unique_everseen(seq)))
        [(), (1,), (0,), (1, 0)]

    """
    s = list(iterable)
    return chain.from_iterable((combinations(s, r) for r in range(len(s) + 1)))
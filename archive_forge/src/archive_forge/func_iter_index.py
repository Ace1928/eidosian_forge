import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def iter_index(iterable, value, start=0):
    """Yield the index of each place in *iterable* that *value* occurs,
    beginning with index *start*.

    See :func:`locate` for a more general means of finding the indexes
    associated with particular values.

    >>> list(iter_index('AABCADEAF', 'A'))
    [0, 1, 4, 7]
    """
    try:
        seq_index = iterable.index
    except AttributeError:
        it = islice(iterable, start, None)
        for i, element in enumerate(it, start):
            if element is value or element == value:
                yield i
    else:
        i = start - 1
        try:
            while True:
                i = seq_index(value, i + 1)
                yield i
        except ValueError:
            pass
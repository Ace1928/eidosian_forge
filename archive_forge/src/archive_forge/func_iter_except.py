import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def iter_except(func, exception, first=None):
    """Yields results from a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like ``iter(func, sentinel)``, but uses an exception instead of a sentinel
    to end the loop.

        >>> l = [0, 1, 2]
        >>> list(iter_except(l.pop, IndexError))
        [2, 1, 0]

    Multiple exceptions can be specified as a stopping condition:

        >>> l = [1, 2, 3, '...', 4, 5, 6]
        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
        [7, 6, 5]
        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
        [4, 3, 2]
        >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
        []

    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass
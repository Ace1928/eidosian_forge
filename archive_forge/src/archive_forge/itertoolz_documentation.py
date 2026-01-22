import itertools
import heapq
import collections
import operator
from functools import partial
from itertools import filterfalse, zip_longest
from collections.abc import Sequence
from toolz.utils import no_default
 Return elements from a sequence with probability of prob

    Returns a lazy iterator of random items from seq.

    ``random_sample`` considers each item independently and without
    replacement. See below how the first time it returned 13 items and the
    next time it returned 6 items.

    >>> seq = list(range(100))
    >>> list(random_sample(0.1, seq)) # doctest: +SKIP
    [6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95]
    >>> list(random_sample(0.1, seq)) # doctest: +SKIP
    [6, 44, 54, 61, 69, 94]

    Providing an integer seed for ``random_state`` will result in
    deterministic sampling. Given the same seed it will return the same sample
    every time.

    >>> list(random_sample(0.1, seq, random_state=2016))
    [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    >>> list(random_sample(0.1, seq, random_state=2016))
    [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]

    ``random_state`` can also be any object with a method ``random`` that
    returns floats between 0.0 and 1.0 (exclusive).

    >>> from random import Random
    >>> randobj = Random(2016)
    >>> list(random_sample(0.1, seq, random_state=randobj))
    [7, 9, 19, 25, 30, 32, 34, 48, 59, 60, 81, 98]
    
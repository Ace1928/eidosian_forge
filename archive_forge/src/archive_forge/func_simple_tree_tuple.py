import functools
import heapq
import itertools
import random
from collections import Counter, OrderedDict, defaultdict
import numpy as np
from . import helpers
def simple_tree_tuple(seq):
    """Make a simple left to right binary tree out of iterable ``seq``.

        >>> tuple_nest([1, 2, 3, 4])
        (((1, 2), 3), 4)

    """
    return functools.reduce(lambda x, y: (x, y), seq)
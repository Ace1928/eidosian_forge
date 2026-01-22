import math
import numbers
import random
import sys
from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from functools import reduce
from operator import mul
from collections import Counter, namedtuple, defaultdict
def fmean(data, weights=None):
    """Convert data to floats and compute the arithmetic mean.

    This runs faster than the mean() function and it always returns a float.
    If the input dataset is empty, it raises a StatisticsError.

    >>> fmean([3.5, 4.0, 5.25])
    4.25
    """
    try:
        n = len(data)
    except TypeError:
        n = 0

        def count(iterable):
            nonlocal n
            for n, x in enumerate(iterable, start=1):
                yield x
        data = count(data)
    if weights is None:
        total = fsum(data)
        if not n:
            raise StatisticsError('fmean requires at least one data point')
        return total / n
    try:
        num_weights = len(weights)
    except TypeError:
        weights = list(weights)
        num_weights = len(weights)
    num = fsum(map(mul, data, weights))
    if n != num_weights:
        raise StatisticsError('data and weights must be the same length')
    den = fsum(weights)
    if not den:
        raise StatisticsError('sum of weights must be non-zero')
    return num / den
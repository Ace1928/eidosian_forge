import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def polynomial_from_roots(roots):
    """Compute a polynomial's coefficients from its roots.

    >>> roots = [5, -4, 3]  # (x - 5) * (x + 4) * (x - 3)
    >>> polynomial_from_roots(roots)  # x^3 - 4 * x^2 - 17 * x + 60
    [1, -4, -17, 60]
    """
    prod = getattr(math, 'prod', lambda x: reduce(operator.mul, x, 1))
    roots = list(map(operator.neg, roots))
    return [sum(map(prod, combinations(roots, k))) for k in range(len(roots) + 1)]
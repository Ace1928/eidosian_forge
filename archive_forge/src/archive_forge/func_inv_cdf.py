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
def inv_cdf(self, p):
    """Inverse cumulative distribution function.  x : P(X <= x) = p

        Finds the value of the random variable such that the probability of
        the variable being less than or equal to that value equals the given
        probability.

        This function is also called the percent point function or quantile
        function.
        """
    if p <= 0.0 or p >= 1.0:
        raise StatisticsError('p must be in the range 0.0 < p < 1.0')
    if self._sigma <= 0.0:
        raise StatisticsError('cdf() not defined when sigma at or below zero')
    return _normal_dist_inv_cdf(p, self._mu, self._sigma)
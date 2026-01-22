from .sage_helper import _within_sage
from functools import reduce
import operator
def correct_max(l):
    """
    Analogous to correct_min.
    """
    are_intervals = [is_RealIntervalFieldElement(x) for x in l]
    if any(are_intervals):
        if not all(are_intervals):
            raise TypeError('Trying to compute max of array where some elements are intervals and others are not.')
        for x in l:
            if x.is_NaN():
                raise ValueError('Trying to compute max of array containing NaN interval.')
        return reduce(lambda x, y: x.max(y), l)
    else:
        for x in l:
            if not x == x:
                raise ValueError('Trying to compute max of array containing NaN.')
        return max(l)
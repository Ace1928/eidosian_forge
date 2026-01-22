import math
import functools
import logging
def roundFunc(tolerance, round=otRound):
    if tolerance < 0:
        raise ValueError('Rounding tolerance must be positive')
    if tolerance == 0:
        return noRound
    if tolerance >= 0.5:
        return round
    return functools.partial(maybeRound, tolerance=tolerance, round=round)
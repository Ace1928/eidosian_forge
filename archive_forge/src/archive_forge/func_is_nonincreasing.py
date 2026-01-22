import operator
import itertools
from pyomo.common.dependencies import numpy, numpy_available, scipy, scipy_available
def is_nonincreasing(vals):
    """Checks if a list of points is nonincreasing"""
    if len(vals) <= 1:
        return True
    it = iter(vals)
    next(it)
    op = operator.le
    return all(itertools.starmap(op, zip(it, vals)))
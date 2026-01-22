import math
import logging
from pyomo.common.errors import InfeasibleConstraintException, IntervalException
def interval_abs(xl, xu):
    abs_xl = abs(xl)
    abs_xu = abs(xu)
    if xl <= 0 and 0 <= xu:
        res_lb = 0
        res_ub = max(abs_xl, abs_xu)
    else:
        res_lb = min(abs_xl, abs_xu)
        res_ub = max(abs_xl, abs_xu)
    return (res_lb, res_ub)
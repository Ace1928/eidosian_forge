from functools import reduce
from sympy.plotting.intervalmath import interval
from sympy.external import import_module
def imax(*args):
    """Evaluates the maximum of a list of intervals"""
    np = import_module('numpy')
    if not all((isinstance(arg, (int, float, interval)) for arg in args)):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float)) or a.is_valid]
        if len(new_args) == 0:
            if all((a.is_valid is False for a in args)):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start for a in new_args]
        end_array = [a if isinstance(a, (int, float)) else a.end for a in new_args]
        return interval(max(start_array), max(end_array))
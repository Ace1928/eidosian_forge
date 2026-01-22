from . import matrix
from deprecated import deprecated
import numbers
import warnings
def check_between(v_min, v_max, **params):
    """Checks parameters are in a specified range

    Parameters
    ----------

    v_min : float, minimum allowed value (inclusive)

    v_max : float, maximum allowed value (inclusive)

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    check_greater(v_min, v_max=v_max)
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            raise ValueError('Expected {} between {} and {}, got {}'.format(p, v_min, v_max, params[p]))
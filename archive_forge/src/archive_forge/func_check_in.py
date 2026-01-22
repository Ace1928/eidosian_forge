from . import matrix
from deprecated import deprecated
import numbers
import warnings
def check_in(choices, **params):
    """Checks parameters are in a list of allowed parameters

    Parameters
    ----------

    choices : array-like, accepted values

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] not in choices:
            raise ValueError('{} value {} not recognized. Choose from {}'.format(p, params[p], choices))
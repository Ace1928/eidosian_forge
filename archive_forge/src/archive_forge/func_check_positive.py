from . import matrix
from deprecated import deprecated
import numbers
import warnings
def check_positive(**params):
    """Check that parameters are positive as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    return check_greater(0, **params)
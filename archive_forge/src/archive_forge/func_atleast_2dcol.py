import numpy as np
from scipy import stats
def atleast_2dcol(x):
    """ convert array_like to 2d from 1d or 0d

    not tested because not used
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 0:
        x = np.atleast_2d(x)
    elif x.ndim > 0:
        raise ValueError('too many dimensions')
    return x
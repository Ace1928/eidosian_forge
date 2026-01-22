import numpy as np
from numpy.linalg import lstsq

    The bisquare function applied to a numpy array.
    The bisquare function is (1-t**2)**2.

    Parameters
    ----------
    t : ndarray
        array bisquare function is applied to, element-wise and in-place.

    Returns
    -------
    Nothing
    
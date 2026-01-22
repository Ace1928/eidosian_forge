from numpy.testing import assert_equal
import numpy as np
def contrast_diff_mean(nm):
    """contrast or restriction matrix for all against mean comparison

    Parameters
    ----------
    nm : int

    Returns
    -------
    contr : ndarray, 2d, (nm-1, nm)
       contrast matrix for all against mean comparisons

    """
    return np.eye(nm) - np.ones((nm, nm)) / nm
from numpy.testing import assert_equal
import numpy as np
def contrast_all_one(nm):
    """contrast or restriction matrix for all against first comparison

    Parameters
    ----------
    nm : int

    Returns
    -------
    contr : ndarray, 2d, (nm-1, nm)
       contrast matrix for all against first comparisons

    """
    contr = np.column_stack((np.ones(nm - 1), -np.eye(nm - 1)))
    return contr
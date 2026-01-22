import numpy as np
from statsmodels.tools.tools import Bunch
MANOVA statistics based on canonical correlation coefficient

    Calculates Pillai's Trace, Wilk's Lambda, Hotelling's Trace and
    Roy's Largest Root.

    Parameters
    ----------
    x1, x2 : ndarrays, 2_D
        two 2-dimensional data arrays, observations in rows, variables in columns
    demean : bool
         If demean is true, then the mean is subtracted from each variable.

    Returns
    -------
    res : dict
        Dictionary containing the test statistics.

    Notes
    -----

    same as `canon` in Stata

    missing: F-statistics and p-values

    TODO: should return a results class instead
    produces nans sometimes, singular, perfect correlation of x1, x2 ?

    
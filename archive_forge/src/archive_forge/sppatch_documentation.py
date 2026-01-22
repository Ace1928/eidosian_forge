from statsmodels.compat.python import lmap
import numpy as np
from scipy import stats, optimize, integrate
calculate and print(Bootstrap or Monte Carlo result

    Parameters
    ----------
    sample : ndarray
        original sample data
    arg : float   (for general case will be array)
    bres : ndarray
        parameter estimates from Bootstrap or Monte Carlo run
    kind : {'bootstrap', 'montecarlo'}
        output is printed for Mootstrap (default) or Monte Carlo

    Returns
    -------
    None, currently only printing

    Notes
    -----
    still a bit a mess because it is used for both Bootstrap and Monte Carlo

    made correction:
        reference point for bootstrap is estimated parameter

    not clear:
        I'm not doing any ddof adjustment in estimation of variance, do we
        need ddof>0 ?

    todo: return results and string instead of printing

    
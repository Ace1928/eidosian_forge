import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def f2_to_wellek(f2, n_groups):
    """Convert Cohen's f-squared to Wellek's effect size (sqrt)

    This computes the following effect size :

       eps = sqrt(n_groups * f2)

    Parameters
    ----------
    f2 : float or ndarray
        Effect size Cohen's f-squared
    n_groups : int
        Number of groups in oneway comparison

    Returns
    -------
    eps : float or ndarray
        Wellek's effect size used in anova equivalence test
    """
    eps = np.sqrt(n_groups * f2)
    return eps
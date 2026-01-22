import numpy as np
from scipy import stats
from scipy.special import ncfdtrinc
from statsmodels.stats.power import ncf_cdf, ncf_ppf
from statsmodels.stats.robust_compare import TrimmedMean, scale_transform
from statsmodels.tools.testing import Holder
from statsmodels.stats.base import HolderTuple
def convert_effectsize_fsqu(f2=None, eta2=None):
    """Convert squared effect sizes in f family

    f2 is signal to noise ratio, var_explained / var_residual

    eta2 is proportion of explained variance, var_explained / var_total

    uses the relationship:
    f2 = eta2 / (1 - eta2)

    Parameters
    ----------
    f2 : None or float
       Squared Cohen's F effect size. If f2 is not None, then eta2 will be
       computed.
    eta2 : None or float
       Squared eta effect size. If f2 is None and eta2 is not None, then f2 is
       computed.

    Returns
    -------
    res : Holder instance
        An instance of the Holder class with f2 and eta2 as attributes.

    """
    if f2 is not None:
        eta2 = 1 / (1 + 1 / f2)
    elif eta2 is not None:
        f2 = eta2 / (1 - eta2)
    res = Holder(f2=f2, eta2=eta2)
    return res
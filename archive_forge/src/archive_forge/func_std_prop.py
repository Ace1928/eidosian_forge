from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def std_prop(prop, nobs):
    """
    Standard error for the estimate of a proportion

    This is just ``np.sqrt(p * (1. - p) / nobs)``

    Parameters
    ----------
    prop : array_like
        proportion
    nobs : int, array_like
        number of observations

    Returns
    -------
    std : array_like
        standard error for a proportion of nobs independent observations
    """
    return np.sqrt(prop * (1.0 - prop) / nobs)
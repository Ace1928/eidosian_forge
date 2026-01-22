from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def linear_harvey_collier(res, order_by=None, skip=None):
    """
    Harvey Collier test for linearity

    The Null hypothesis is that the regression is correctly modeled as linear.

    Parameters
    ----------
    res : RegressionResults
        A results instance from a linear regression.
    order_by : array_like, default None
        Integer array specifying the order of the residuals. If not provided,
        the order of the residuals is not changed. If provided, must have
        the same number of observations as the endogenous variable.
    skip : int, default None
        The number of observations to use for initial OLS, if None then skip is
        set equal to the number of regressors (columns in exog).

    Returns
    -------
    tvalue : float
        The test statistic, based on ttest_1sample.
    pvalue : float
        The pvalue of the test.

    See Also
    --------
    statsmodels.stats.diadnostic.recursive_olsresiduals
        Recursive OLS residual calculation used in the test.

    Notes
    -----
    This test is a t-test that the mean of the recursive ols residuals is zero.
    Calculating the recursive residuals might take some time for large samples.
    """
    rr = recursive_olsresiduals(res, skip=skip, alpha=0.95, order_by=order_by)
    return stats.ttest_1samp(rr[3][3:], 0)
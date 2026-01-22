import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_chi2_mixed():
    """
    Test fitting just the shape parameter (df) of chi2 to mixed data.

    Calculation in R, with:
    * two values not censored [1, 10],
    * one left-censored [1],
    * one right-censored [30], and
    * one interval-censored [[4, 8]].

    > library(fitdistrplus)
    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))
    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' chisq ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    df 5.060329
    """
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30], interval=[[4, 8]])
    df, loc, scale = chi2.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(df, 5.060329, rtol=5e-06)
    assert loc == 0
    assert scale == 1
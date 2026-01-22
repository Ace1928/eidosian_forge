import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_cauchy_mixed():
    """
    Test fitting the Cauchy distribution to data with mixed censoring.

    Calculation in R, with:
    * two values not censored [1, 10],
    * one left-censored [1],
    * one right-censored [30], and
    * one interval-censored [[4, 8]].

    > library(fitdistrplus)
    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))
    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' cauchy ' on censored data by maximum
    likelihood
    Parameters:
             estimate
    location 4.605150
    scale    5.900852
    """
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30], interval=[[4, 8]])
    loc, scale = cauchy.fit(data, optimizer=optimizer)
    assert_allclose(loc, 4.60515, rtol=5e-06)
    assert_allclose(scale, 5.900852, rtol=5e-06)
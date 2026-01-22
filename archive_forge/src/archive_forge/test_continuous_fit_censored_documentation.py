import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,

    Test fitting the normal distribution to interval-censored data.

    Calculation in R:

    > library(fitdistrplus)
    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),
    +                    right=c(0.20, 0.55, 0.90, 0.95))
    > result = fitdistcens(data, 'norm', control=list(reltol=1e-14))

    > result
    Fitting of the distribution ' norm ' on censored data by maximum likelihood
    Parameters:
          estimate
    mean 0.5919990
    sd   0.2868042
    > result$sd
         mean        sd
    0.1444432 0.1029451
    
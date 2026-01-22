import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import (CensoredData, beta, cauchy, chi2, expon, gamma,
def test_expon_right_censored():
    """
    For the exponential distribution with loc=0, the exact solution for
    fitting n uncensored points x[0]...x[n-1] and m right-censored points
    x[n]..x[n+m-1] is

        scale = sum(x)/n

    That is, divide the sum of all the values (not censored and
    right-censored) by the number of uncensored values.  (See, for example,
    https://en.wikipedia.org/wiki/Censoring_(statistics)#Likelihood.)

    The second derivative of the log-likelihood function is

        n/scale**2 - 2*sum(x)/scale**3

    from which the estimate of the standard error can be computed.

    -----

    Calculation in R, for reference only. The R results are not
    used in the test.

    > library(fitdistrplus)
    > dexps <- function(x, scale) {
    +     return(dexp(x, 1/scale))
    + }
    > pexps <- function(q, scale) {
    +     return(pexp(q, 1/scale))
    + }
    > left <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,
    +                                     16, 16, 20, 20, 21, 22)
    > right <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,
    +                                     NA, NA, NA, NA, NA, NA)
    > result = fitdistcens(data, 'exps', start=list(scale=mean(data$left)),
    +                      control=list(reltol=1e-14))
    > result
    Fitting of the distribution ' exps ' on censored data by maximum likelihood
    Parameters:
          estimate
    scale    19.85
    > result$sd
       scale
    6.277119
    """
    obs = [1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15, 16, 16, 20, 20, 21, 22]
    cens = [False] * 10 + [True] * 6
    data = CensoredData.right_censored(obs, cens)
    loc, scale = expon.fit(data, floc=0, optimizer=optimizer)
    assert loc == 0
    n = len(data) - data.num_censored()
    total = data._uncensored.sum() + data._right.sum()
    expected = total / n
    assert_allclose(scale, expected, 1e-08)
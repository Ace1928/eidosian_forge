import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi
from .extras import mvstdnormcdf
from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln

        > lower <- -1
        > upper <- 3
        > df <- 4
        > corr <- diag(3)
        > delta <- rep(0, 3)
        > pmvt(lower=lower, upper=upper, delta=delta, df=df, corr=corr)
        [1] 0.5300413
        attr(,"error")
        [1] 4.321136e-05
        attr(,"msg")
        [1] "Normal Completion"
        > (pt(upper, df) - pt(lower, df))**3
        [1] 0.4988254

    
import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.sppatch import expect_v2
from .distparams import distcont
def mc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis
    """
    mc, mc2, mc3, mc4 = args
    skew = np.divide(mc3, mc2 ** 1.5)
    kurt = np.divide(mc4, mc2 ** 2.0) - 3.0
    return (mc, mc2, skew, kurt)
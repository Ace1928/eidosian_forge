import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ncf_cdf(x, dfn, dfd, nc):
    return special.ncfdtr(dfn, dfd, nc, x)
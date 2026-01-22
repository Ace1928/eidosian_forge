import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def ncf_ppf(q, dfn, dfd, nc):
    return special.ncfdtri(dfn, dfd, nc, q)
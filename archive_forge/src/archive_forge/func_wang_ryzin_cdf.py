import numpy as np
from scipy.special import erf
def wang_ryzin_cdf(h, Xi, x_u):
    ordered = np.zeros(Xi.size)
    for x in np.unique(Xi):
        if x <= x_u:
            ordered += wang_ryzin(h, Xi, x)
    return ordered
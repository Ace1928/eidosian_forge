import numpy as np
from .copulas import Copula
def copula_bv_ev(u, transform, args=()):
    """generic bivariate extreme value copula
    """
    u, v = u
    return np.exp(np.log(u * v) * transform(np.log(u) / np.log(u * v), *args))
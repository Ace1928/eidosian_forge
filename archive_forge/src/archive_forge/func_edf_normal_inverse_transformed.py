import numpy as np
from numpy import ma
from scipy import stats
def edf_normal_inverse_transformed(x, alpha=3.0 / 8, beta=3.0 / 8, axis=0):
    """rank based normal inverse transformed cdf
    """
    from scipy import stats
    ranks = plotting_positions(x, alpha=alpha, beta=alpha, axis=0, masknan=False)
    ranks_transf = stats.norm.ppf(ranks)
    return ranks_transf
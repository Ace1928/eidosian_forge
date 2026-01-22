import numpy as np
from scipy import special
from scipy.special import gammaln
def mean_grad(x, beta):
    """gradient/Jacobian for d (x*beta)/ d beta
    """
    return x
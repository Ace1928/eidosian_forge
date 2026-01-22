import numpy as np
from scipy import special
from scipy.special import gammaln
def norm_dlldy(y):
    """derivative of log pdf of standard normal with respect to y
    """
    return -y
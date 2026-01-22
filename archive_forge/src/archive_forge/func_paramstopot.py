import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
def paramstopot(thresh, shape, scale):
    """transform shape scale for peak over threshold

    y = x-u|x>u ~ GPD(k, sigma-k*u) if x ~ GPD(k, sigma)
    notation of de Zea Bermudez, Kotz
    k, sigma is shape, scale
    """
    return (shape, scale - shape * thresh)
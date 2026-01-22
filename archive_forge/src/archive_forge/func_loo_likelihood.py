import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from . import kernels
def loo_likelihood(self):
    raise NotImplementedError
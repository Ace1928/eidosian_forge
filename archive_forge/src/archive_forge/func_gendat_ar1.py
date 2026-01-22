from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
def gendat_ar1():
    ars = AR_simulator()
    ars.ngroups = 200
    ars.params = np.r_[0, -0.8, 1.2, 0, 0.5]
    ars.error_sd = 2
    ars.dparams = [ar]
    ars.simulate()
    return (ars, Autoregressive())
import math
import numpy as np
from scipy import linalg, stats, special
from .linalg_decomp_1 import SvdArray
loglike multivariate normal

    assumes x is 1d, (nobs,) and sigma is 2d (nobs, nobs)

    brute force from formula
    no checking of correct inputs
    use of inv and log-det should be replace with something more efficient
    
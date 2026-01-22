from __future__ import absolute_import
import scipy.special
import autograd.numpy as np
from autograd.extend import primitive, defvjp, defjvp
from autograd.numpy.numpy_vjps import unbroadcast_f, repeat_to_match_shape
def gammainc_vjp_arg1(ans, a, x):
    coeffs = sign * np.exp(-x) * np.power(x, a - 1) / gamma(a)
    return unbroadcast_f(x, lambda g: g * coeffs)
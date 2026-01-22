from __future__ import absolute_import
import autograd.numpy as np
import scipy.stats
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.scipy.special import gamma, psi
def grad_gamma_logpdf_arg1(x, a):
    return np.log(x) - psi(a)
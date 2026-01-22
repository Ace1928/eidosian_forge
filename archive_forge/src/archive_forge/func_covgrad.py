from __future__ import absolute_import
import scipy.stats
import autograd.numpy as np
from autograd.numpy.numpy_vjps import unbroadcast_f
from autograd.extend import primitive, defvjp
def covgrad(x, mean, cov, allow_singular=False):
    if allow_singular:
        raise NotImplementedError('The multivariate normal pdf is not differentiable w.r.t. a singular covariance matix')
    J = np.linalg.inv(cov)
    solved = np.matmul(J, np.expand_dims(x - mean, -1))
    return 1.0 / 2 * (generalized_outer_product(solved) - J)
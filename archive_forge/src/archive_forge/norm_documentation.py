from __future__ import absolute_import
import scipy.stats
import autograd.numpy as anp
from autograd.extend import primitive, defvjp
from autograd.numpy.numpy_vjps import unbroadcast_f
Gradients of the normal distribution.
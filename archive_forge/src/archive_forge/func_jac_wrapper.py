import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def jac_wrapper(t, y):
    jac = asarray(jacfunc(t, y, *jac_params))
    padded_jac = vstack((jac, zeros((ml, jac.shape[1]))))
    return padded_jac
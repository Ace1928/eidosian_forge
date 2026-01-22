import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def set_jac_params(self, *args):
    """Set extra parameters for user-supplied function jac."""
    self.jac_params = args
    return self
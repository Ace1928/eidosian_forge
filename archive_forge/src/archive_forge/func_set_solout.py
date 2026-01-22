import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def set_solout(self, solout, complex=False):
    self.solout = solout
    self.solout_cmplx = complex
    if solout is None:
        self.iout = 0
    else:
        self.iout = 1
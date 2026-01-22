import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
def find_integrator(name):
    for cl in IntegratorBase.integrator_classes:
        if re.match(name, cl.__name__, re.I):
            return cl
    return None
from __future__ import division
from builtins import next
from builtins import zip
from builtins import range
import sys
import inspect
import numpy
from numpy.core import numeric
import uncertainties.umath_core as umath_core
import uncertainties.core as uncert_core
from uncertainties.core import deprecation
def umatrix(nominal_values, std_devs=None):
    """
    Constructs a matrix that contains numbers with uncertainties.

    The arguments are the same as for uarray(...): nominal values, and
    standard deviations.

    The returned matrix can be inverted, thanks to the fact that it is
    a unumpy.matrix object instead of a numpy.matrix one.
    """
    if std_devs is None:
        deprecation('umatrix() should now be called with two arguments.')
        nominal_values, std_devs = nominal_values
    return uarray(nominal_values, std_devs).view(matrix)
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
def inv_with_derivatives(arr, input_type, derivatives):
    """
    Defines the matrix inverse and its derivatives.

    See the definition of func_with_deriv_to_uncert_func() for its
    detailed semantics.
    """
    inverse = numpy.linalg.inv(arr)
    if issubclass(input_type, numpy.matrix):
        inverse = inverse.view(numpy.matrix)
    yield inverse
    inverse_mat = numpy.asmatrix(inverse)
    for derivative in derivatives:
        derivative_mat = numpy.asmatrix(derivative)
        yield (-inverse_mat * derivative_mat * inverse_mat)
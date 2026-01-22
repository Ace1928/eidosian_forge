from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
def correlated_values(nom_values, covariance_mat, tags=None):
    """
        Return numbers with uncertainties (AffineScalarFunc objects)
        that correctly reproduce the given covariance matrix, and have
        the given (float) values as their nominal value.

        The correlated_values_norm() function returns the same result,
        but takes a correlation matrix instead of a covariance matrix.

        The list of values and the covariance matrix must have the
        same length, and the matrix must be a square (symmetric) one.

        The numbers with uncertainties returned depend on newly
        created, independent variables (Variable objects).

        nom_values -- sequence with the nominal (real) values of the
        numbers with uncertainties to be returned.

        covariance_mat -- full covariance matrix of the returned numbers with
        uncertainties. For example, the first element of this matrix is the
        variance of the first number with uncertainty. This matrix must be a
        NumPy array-like (list of lists, NumPy array, etc.). 

        tags -- if 'tags' is not None, it must list the tag of each new
        independent variable.
        """
    std_devs = numpy.sqrt(numpy.diag(covariance_mat))
    norm_vector = std_devs.copy()
    norm_vector[norm_vector == 0] = 1
    return correlated_values_norm(list(zip(nom_values, std_devs)), covariance_mat / norm_vector / norm_vector[:, numpy.newaxis], tags)
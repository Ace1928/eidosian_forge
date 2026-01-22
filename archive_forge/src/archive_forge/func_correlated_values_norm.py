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
def correlated_values_norm(values_with_std_dev, correlation_mat, tags=None):
    """
        Return correlated values like correlated_values(), but takes
        instead as input:

        - nominal (float) values along with their standard deviation, and
        - a correlation matrix (i.e. a normalized covariance matrix).

        values_with_std_dev -- sequence of (nominal value, standard
        deviation) pairs. The returned, correlated values have these
        nominal values and standard deviations.

        correlation_mat -- correlation matrix between the given values, except
        that any value with a 0 standard deviation must have its correlations
        set to 0, with a diagonal element set to an arbitrary value (something
        close to 0-1 is recommended, for a better numerical precision).  When
        no value has a 0 variance, this is the covariance matrix normalized by
        standard deviations, and thus a symmetric matrix with ones on its
        diagonal.  This matrix must be an NumPy array-like (list of lists,
        NumPy array, etc.).

        tags -- like for correlated_values().
        """
    if tags is None:
        tags = (None,) * len(values_with_std_dev)
    nominal_values, std_devs = numpy.transpose(values_with_std_dev)
    variances, transform = numpy.linalg.eigh(correlation_mat)
    variances[variances < 0] = 0.0
    variables = tuple((Variable(0, sqrt(variance), tag) for variance, tag in zip(variances, tags)))
    transform *= std_devs[:, numpy.newaxis]
    values_funcs = tuple((AffineScalarFunc(value, LinearCombination(dict(zip(variables, coords)))) for coords, value in zip(transform, nominal_values)))
    return values_funcs
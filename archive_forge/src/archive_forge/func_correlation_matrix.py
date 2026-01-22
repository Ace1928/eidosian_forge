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
def correlation_matrix(nums_with_uncert):
    """
        Return the correlation matrix of the given sequence of
        numbers with uncertainties, as a NumPy array of floats.
        """
    cov_mat = numpy.array(covariance_matrix(nums_with_uncert))
    std_devs = numpy.sqrt(cov_mat.diagonal())
    return cov_mat / std_devs / std_devs[numpy.newaxis].T
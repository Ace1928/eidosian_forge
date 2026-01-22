import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
def test_crystalball_function():
    """
    All values are calculated using the independent implementation of the
    ROOT framework (see https://root.cern.ch/).
    Corresponding ROOT code is given in the comments.
    """
    X = np.linspace(-5.0, 5.0, 21)[:-1]
    calculated = stats.crystalball.pdf(X, beta=1.0, m=2.0)
    expected = np.array([0.0202867, 0.0241428, 0.0292128, 0.0360652, 0.045645, 0.059618, 0.0811467, 0.116851, 0.18258, 0.265652, 0.301023, 0.265652, 0.18258, 0.097728, 0.0407391, 0.013226, 0.00334407, 0.000658486, 0.000100982, 1.20606e-05])
    assert_allclose(expected, calculated, rtol=0.001)
    calculated = stats.crystalball.pdf(X, beta=2.0, m=3.0)
    expected = np.array([0.0019648, 0.00279754, 0.00417592, 0.00663121, 0.0114587, 0.0223803, 0.0530497, 0.12726, 0.237752, 0.345928, 0.391987, 0.345928, 0.237752, 0.12726, 0.0530497, 0.0172227, 0.00435458, 0.000857469, 0.000131497, 1.57051e-05])
    assert_allclose(expected, calculated, rtol=0.001)
    calculated = stats.crystalball.pdf(X, beta=2.0, m=3.0, loc=0.5, scale=2.0)
    expected = np.array([0.00785921, 0.0111902, 0.0167037, 0.0265249, 0.0423866, 0.0636298, 0.0897324, 0.118876, 0.147944, 0.172964, 0.189964, 0.195994, 0.189964, 0.172964, 0.147944, 0.118876, 0.0897324, 0.0636298, 0.0423866, 0.0265249])
    assert_allclose(expected, calculated, rtol=0.001)
    calculated = stats.crystalball.cdf(X, beta=1.0, m=2.0)
    expected = np.array([0.12172, 0.132785, 0.146064, 0.162293, 0.18258, 0.208663, 0.24344, 0.292128, 0.36516, 0.478254, 0.622723, 0.767192, 0.880286, 0.94959, 0.982834, 0.995314, 0.998981, 0.999824, 0.999976, 0.999997])
    assert_allclose(expected, calculated, rtol=0.001)
    calculated = stats.crystalball.cdf(X, beta=2.0, m=3.0)
    expected = np.array([0.00442081, 0.00559509, 0.00730787, 0.00994682, 0.0143234, 0.0223803, 0.0397873, 0.0830763, 0.173323, 0.320592, 0.508717, 0.696841, 0.844111, 0.934357, 0.977646, 0.993899, 0.998674, 0.999771, 0.999969, 0.999997])
    assert_allclose(expected, calculated, rtol=0.001)
    calculated = stats.crystalball.cdf(X, beta=2.0, m=3.0, loc=0.5, scale=2.0)
    expected = np.array([0.0176832, 0.0223803, 0.0292315, 0.0397873, 0.0567945, 0.0830763, 0.121242, 0.173323, 0.24011, 0.320592, 0.411731, 0.508717, 0.605702, 0.696841, 0.777324, 0.844111, 0.896192, 0.934357, 0.960639, 0.977646])
    assert_allclose(expected, calculated, rtol=0.001)
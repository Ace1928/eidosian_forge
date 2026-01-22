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
def test_levy_cdf_ppf():
    x = np.array([1000, 1.0, 0.5, 0.1, 0.01, 0.001])
    expected = np.array([0.9747728793699604, 0.3173105078629141, 0.1572992070502851, 0.0015654022580025495, 1.523970604832105e-23, 1.795832784800726e-219])
    y = stats.levy.cdf(x)
    assert_allclose(y, expected, rtol=1e-10)
    xx = stats.levy.ppf(expected)
    assert_allclose(xx, x, rtol=1e-13)
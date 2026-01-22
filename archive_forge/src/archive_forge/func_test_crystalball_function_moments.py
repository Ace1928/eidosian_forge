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
def test_crystalball_function_moments():
    """
    All values are calculated using the pdf formula and the integrate function
    of Mathematica
    """
    beta = np.array([2.0, 1.0, 3.0, 2.0, 3.0])
    m = np.array([3.0, 3.0, 2.0, 4.0, 9.0])
    expected_0th_moment = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    calculated_0th_moment = stats.crystalball._munp(0, beta, m)
    assert_allclose(expected_0th_moment, calculated_0th_moment, rtol=0.001)
    norm = np.array([2.5511, 3.01873, 2.51065, 2.53983, 2.507410455])
    a = np.array([-0.21992, -3.03265, np.inf, -0.135335, -0.003174])
    expected_1th_moment = a / norm
    calculated_1th_moment = stats.crystalball._munp(1, beta, m)
    assert_allclose(expected_1th_moment, calculated_1th_moment, rtol=0.001)
    a = np.array([np.inf, np.inf, np.inf, 3.2616, 2.519908])
    expected_2th_moment = a / norm
    calculated_2th_moment = stats.crystalball._munp(2, beta, m)
    assert_allclose(expected_2th_moment, calculated_2th_moment, rtol=0.001)
    a = np.array([np.inf, np.inf, np.inf, np.inf, -0.0577668])
    expected_3th_moment = a / norm
    calculated_3th_moment = stats.crystalball._munp(3, beta, m)
    assert_allclose(expected_3th_moment, calculated_3th_moment, rtol=0.001)
    a = np.array([np.inf, np.inf, np.inf, np.inf, 7.78468])
    expected_4th_moment = a / norm
    calculated_4th_moment = stats.crystalball._munp(4, beta, m)
    assert_allclose(expected_4th_moment, calculated_4th_moment, rtol=0.001)
    a = np.array([np.inf, np.inf, np.inf, np.inf, -1.31086])
    expected_5th_moment = a / norm
    calculated_5th_moment = stats.crystalball._munp(5, beta, m)
    assert_allclose(expected_5th_moment, calculated_5th_moment, rtol=0.001)
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
def test_burr_nan_mean_var_9544(self):
    c, d = (0.5, 3)
    mean, variance = stats.burr(c, d).stats()
    assert_(np.isnan(mean))
    assert_(np.isnan(variance))
    c, d = (1.5, 3)
    mean, variance = stats.burr(c, d).stats()
    assert_(np.isfinite(mean))
    assert_(np.isnan(variance))
    c, d = (0.5, 3)
    e1, e2, e3, e4 = stats.burr._munp(np.array([1, 2, 3, 4]), c, d)
    assert_(np.isnan(e1))
    assert_(np.isnan(e2))
    assert_(np.isnan(e3))
    assert_(np.isnan(e4))
    c, d = (1.5, 3)
    e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
    assert_(np.isfinite(e1))
    assert_(np.isnan(e2))
    assert_(np.isnan(e3))
    assert_(np.isnan(e4))
    c, d = (2.5, 3)
    e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
    assert_(np.isfinite(e1))
    assert_(np.isfinite(e2))
    assert_(np.isnan(e3))
    assert_(np.isnan(e4))
    c, d = (3.5, 3)
    e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
    assert_(np.isfinite(e1))
    assert_(np.isfinite(e2))
    assert_(np.isfinite(e3))
    assert_(np.isnan(e4))
    c, d = (4.5, 3)
    e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
    assert_(np.isfinite(e1))
    assert_(np.isfinite(e2))
    assert_(np.isfinite(e3))
    assert_(np.isfinite(e4))
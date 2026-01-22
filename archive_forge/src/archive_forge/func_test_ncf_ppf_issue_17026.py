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
@pytest.mark.skipif(sys.maxsize <= 2 ** 32, reason='On some 32-bit the warning is not raised')
def test_ncf_ppf_issue_17026():
    x = np.linspace(0, 1, 600)
    x[0] = 1e-16
    par = (0.1, 2, 5, 0, 1)
    with pytest.warns(RuntimeWarning):
        q = stats.ncf.ppf(x, *par)
        q0 = [stats.ncf.ppf(xi, *par) for xi in x]
    assert_allclose(q, q0)
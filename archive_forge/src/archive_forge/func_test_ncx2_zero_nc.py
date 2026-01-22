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
@pytest.mark.parametrize('method, expected', [('cdf', np.array([2.497951336e-09, 3.437288941e-10])), ('pdf', np.array([1.23857998e-07, 1.710041145e-08])), ('logpdf', np.array([-15.90413011, -17.88416331])), ('ppf', np.array([4.865182052, 7.017182271]))])
def test_ncx2_zero_nc(method, expected):
    result = getattr(stats.ncx2, method)(0.1, nc=[0, 4], df=10)
    assert_allclose(result, expected, atol=1e-15)
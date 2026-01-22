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
@pytest.mark.parametrize('x, a, b, sf, rtol', [(-1, 1, 0, 0.8759652211005315, 1e-13), (25, 1, 0, 1.1318690184042579e-13, 0.0001), (1, 5, -1.5, 0.002066711134653577, 1e-12), (10, 5, -1.5, 2.308435233930669e-29, 1e-09)])
def test_sf_isf_mpmath(self, x, a, b, sf, rtol):
    s = stats.norminvgauss.sf(x, a, b)
    assert_allclose(s, sf, rtol=rtol)
    i = stats.norminvgauss.isf(sf, a, b)
    assert_allclose(i, x, rtol=rtol)
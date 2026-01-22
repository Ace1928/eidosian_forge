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
def test_sf_isf_mpmath_vectorized(self):
    x = [-1, 25]
    a = [1, 1]
    b = 0
    sf = [0.8759652211005315, 1.1318690184042579e-13]
    s = stats.norminvgauss.sf(x, a, b)
    assert_allclose(s, sf, rtol=1e-13, atol=1e-16)
    i = stats.norminvgauss.isf(sf, a, b)
    assert_allclose(i, x, rtol=1e-06)
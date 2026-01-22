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
def test_hypergeom(self):
    m_true, v_true = stats.hypergeom.stats(20, 10, 8, loc=5.0)
    m = stats.hypergeom.expect(lambda x: x, args=(20, 10, 8), loc=5.0)
    assert_almost_equal(m, m_true, decimal=13)
    v = stats.hypergeom.expect(lambda x: (x - 9.0) ** 2, args=(20, 10, 8), loc=5.0)
    assert_almost_equal(v, v_true, decimal=14)
    v_bounds = stats.hypergeom.expect(lambda x: (x - 9.0) ** 2, args=(20, 10, 8), loc=5.0, lb=5, ub=13)
    assert_almost_equal(v_bounds, v_true, decimal=14)
    prob_true = 1 - stats.hypergeom.pmf([5, 13], 20, 10, 8, loc=5).sum()
    prob_bounds = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), loc=5.0, lb=6, ub=12)
    assert_almost_equal(prob_bounds, prob_true, decimal=13)
    prob_bc = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), loc=5.0, lb=6, ub=12, conditional=True)
    assert_almost_equal(prob_bc, 1, decimal=14)
    prob_b = stats.hypergeom.expect(lambda x: 1, args=(20, 10, 8), lb=0, ub=8)
    assert_almost_equal(prob_b, 1, decimal=13)
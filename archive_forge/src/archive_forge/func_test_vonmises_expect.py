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
def test_vonmises_expect(self):
    """
        Test that the vonmises expectation values are
        computed correctly.  This test checks that the
        numeric integration estimates the correct normalization
        (1) and mean angle (loc).  These expectations are
        independent of the chosen 2pi interval.
        """
    rng = np.random.default_rng(6762668991392531563)
    loc, kappa, lb = rng.random(3) * 10
    res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1)
    assert_allclose(res, 1)
    assert np.issubdtype(res.dtype, np.floating)
    bounds = (lb, lb + 2 * np.pi)
    res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: 1, *bounds)
    assert_allclose(res, 1)
    assert np.issubdtype(res.dtype, np.floating)
    bounds = (lb, lb + 2 * np.pi)
    res = stats.vonmises(loc=loc, kappa=kappa).expect(lambda x: np.exp(1j * x), *bounds, complex_func=1)
    assert_allclose(np.angle(res), loc % (2 * np.pi))
    assert np.issubdtype(res.dtype, np.complexfloating)
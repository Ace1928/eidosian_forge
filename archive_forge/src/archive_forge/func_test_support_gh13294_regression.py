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
@pytest.mark.parametrize('distname, args', invdistdiscrete + invdistcont)
def test_support_gh13294_regression(distname, args):
    if distname in skip_test_support_gh13294_regression:
        pytest.skip(f'skipping test for the support method for distribution {distname}.')
    dist = getattr(stats, distname)
    if isinstance(dist, stats.rv_continuous):
        if len(args) != 0:
            a0, b0 = dist.support(*args)
            assert_equal(a0, np.nan)
            assert_equal(b0, np.nan)
        loc1, scale1 = (0, -1)
        a1, b1 = dist.support(*args, loc1, scale1)
        assert_equal(a1, np.nan)
        assert_equal(b1, np.nan)
    else:
        a, b = dist.support(*args)
        assert_equal(a, np.nan)
        assert_equal(b, np.nan)
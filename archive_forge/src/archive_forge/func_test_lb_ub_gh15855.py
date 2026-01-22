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
def test_lb_ub_gh15855(self):
    dist = stats.uniform
    ref = dist.mean(loc=10, scale=5)
    assert_allclose(dist.expect(loc=10, scale=5), ref)
    assert_allclose(dist.expect(loc=10, scale=5, lb=9, ub=16), ref)
    assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14), ref * 0.6)
    assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=14, conditional=True), ref)
    assert_allclose(dist.expect(loc=10, scale=5, lb=11, ub=13), 12 * 0.4)
    assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11), -12 * 0.4)
    assert_allclose(dist.expect(loc=10, scale=5, lb=13, ub=11, conditional=True), 12)
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
def test_ab(self):
    c = -0.1
    rv = stats.genpareto(c=c)
    a, b = rv.dist._get_support(c)
    assert_equal([a, b], [0.0, 10.0])
    c = 0.1
    stats.genpareto.pdf(0, c=c)
    assert_equal(rv.dist._get_support(c), [0, np.inf])
    c = -0.1
    rv = stats.genpareto(c=c)
    a, b = rv.dist._get_support(c)
    assert_equal([a, b], [0.0, 10.0])
    c = 0.1
    stats.genpareto.pdf(0, c)
    assert_equal((rv.dist.a, rv.dist.b), stats.genpareto._get_support(c))
    rv1 = stats.genpareto(c=0.1)
    assert_(rv1.dist is not rv.dist)
    for c in [1.0, 0.0]:
        c = np.asarray(c)
        rv = stats.genpareto(c=c)
        a, b = (rv.a, rv.b)
        assert_equal(a, 0.0)
        assert_(np.isposinf(b))
        c = np.asarray(-2.0)
        a, b = stats.genpareto._get_support(c)
        assert_allclose([a, b], [0.0, 0.5])
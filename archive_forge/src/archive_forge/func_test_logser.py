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
def test_logser(self):
    p, loc = (0.3, 3)
    res_0 = stats.logser.expect(lambda k: k, args=(p,))
    assert_allclose(res_0, p / (p - 1.0) / np.log(1.0 - p), atol=1e-15)
    res_l = stats.logser.expect(lambda k: k, args=(p,), loc=loc)
    assert_allclose(res_l, res_0 + loc, atol=1e-15)
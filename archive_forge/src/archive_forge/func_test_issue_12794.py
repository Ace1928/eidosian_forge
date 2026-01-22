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
@pytest.mark.skipif(MACOS_INTEL, reason='Overflow, see gh-14901')
def test_issue_12794(self):
    inv_R = np.array([0.0004944464889611935, 0.0018360586912635726, 0.012266391994251835])
    count_list = np.array([10, 100, 1000])
    p = 1e-11
    inv = stats.beta.isf(p, count_list + 1, 100000 - count_list)
    assert_allclose(inv, inv_R)
    res = stats.beta.sf(inv, count_list + 1, 100000 - count_list)
    assert_allclose(res, p)
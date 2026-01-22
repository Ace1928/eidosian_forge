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
@pytest.mark.parametrize('x, p, a, b, loc, scale, ref', [(0, 1e-06, 12, -1, 0, 1, 0.38520358671350524), (-1, 3, 2.5, 2.375, 1, 3, 0.9999901774267577), (-20, 3, 2.5, 2.375, 1, 3, 1.0), (25, 2, 3, 1.5, 0.5, 1.5, 8.593419916523976e-10), (300, 10, 1.5, 0.25, 1, 5, 6.137415609872158e-24), (60, -0.125, 1000, 999, 0, 1, 2.4524915075944173e-12), (75, -0.125, 1000, 999, 0, 1, 2.9435194886214633e-18)])
def test_sf_mpmath(self, x, p, a, b, loc, scale, ref):
    sf = stats.genhyperbolic.sf(x, p, a, b, loc=loc, scale=scale)
    assert_allclose(sf, ref, rtol=5e-12)
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
@pytest.mark.parametrize('x, p, a, b, loc, scale, ref', [(-15, 2, 3, 1.5, 0.5, 1.5, 4.770036428808252e-20), (-15, 10, 1.5, 0.25, 1, 5, 0.03282964575089294), (-15, 10, 1.5, 1.375, 0, 1, 3.3711159600215594e-23), (-15, 0.125, 1.5, 1.49995, 0, 1, 4.729401428898605e-23), (-1, 0.125, 1.5, 1.49995, 0, 1, 0.0003565725914786859), (5, -0.125, 1.5, 1.49995, 0, 1, 0.2600651974023352), (5, -0.125, 1000, 999, 0, 1, 5.923270556517253e-28), (20, -0.125, 1000, 999, 0, 1, 0.23452293711665634), (40, -0.125, 1000, 999, 0, 1, 0.9999648749561968), (60, -0.125, 1000, 999, 0, 1, 0.9999999999975475)])
def test_cdf_mpmath(self, x, p, a, b, loc, scale, ref):
    cdf = stats.genhyperbolic.cdf(x, p, a, b, loc=loc, scale=scale)
    assert_allclose(cdf, ref, rtol=5e-12)
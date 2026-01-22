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
@pytest.mark.parametrize('df1,df2,x', [(2, 2, [-0.5, 0.2, 1.0, 2.3]), (4, 11, [-0.5, 0.2, 1.0, 2.3]), (7, 17, [1, 2, 3, 4, 5])])
def test_ncf_edge_case(df1, df2, x):
    nc = 0
    expected_cdf = stats.f.cdf(x, df1, df2)
    calculated_cdf = stats.ncf.cdf(x, df1, df2, nc)
    assert_allclose(expected_cdf, calculated_cdf, rtol=1e-14)
    expected_pdf = stats.f.pdf(x, df1, df2)
    calculated_pdf = stats.ncf.pdf(x, df1, df2, nc)
    assert_allclose(expected_pdf, calculated_pdf, rtol=1e-06)
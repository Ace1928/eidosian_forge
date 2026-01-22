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
@pytest.mark.xslow
def test_pdf_against_cdf(self):
    k, v = (3, 10)
    x = np.arange(0, 10, step=0.01)
    y_cdf = stats.studentized_range.cdf(x, k, v)[1:]
    y_pdf_raw = stats.studentized_range.pdf(x, k, v)
    y_pdf_cumulative = cumulative_trapezoid(y_pdf_raw, x)
    assert_allclose(y_pdf_cumulative, y_cdf, rtol=0.0001)
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
def test_ppf_bug_17050(self):
    skews = [-3, -1, 0, 0.5]
    x_eval = 0.5
    res = stats.pearson3.ppf(stats.pearson3.cdf(x_eval, skews), skews)
    assert_allclose(res, x_eval)
    skew = np.array([[-0.5], [1.5]])
    x = np.linspace(-2, 2)
    assert_allclose(stats.pearson3.pdf(x, skew), stats.pearson3.pdf(-x, -skew))
    assert_allclose(stats.pearson3.cdf(x, skew), stats.pearson3.sf(-x, -skew))
    assert_allclose(stats.pearson3.ppf(x, skew), -stats.pearson3.isf(x, -skew))
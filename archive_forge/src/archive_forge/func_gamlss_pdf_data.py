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
@pytest.fixture
def gamlss_pdf_data(self):
    """Sample data points computed using the `ST5` distribution from the
        GAMLSS package in R. The pdf has been calculated for (a,b)=(2,3),
        (a,b)=(8,4), and (a,b)=(12,13) for x in `np.linspace(-10, 10, 41)`.

        N.B. the `ST5` distribution in R uses an alternative parameterization
        in terms of nu and tau, where:
            - nu = (a - b) / (a * b * (a + b)) ** 0.5
            - tau = 2 / (a + b)
        """
    data = np.load(Path(__file__).parent / 'data/jf_skew_t_gamlss_pdf_data.npy')
    return np.rec.fromarrays(data, names='x,pdf,a,b')
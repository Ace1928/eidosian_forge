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
def test_invgauss(self):
    ig = stats.geninvgauss.rvs(size=1500, p=-0.5, b=1, random_state=1234)
    assert_equal(stats.kstest(ig, 'invgauss', args=[1])[1] > 0.15, True)
    mu, x = (100, np.linspace(0.01, 1, 10))
    pdf_ig = stats.geninvgauss.pdf(x, p=-0.5, b=1 / mu, scale=mu)
    assert_allclose(pdf_ig, stats.invgauss(mu).pdf(x))
    cdf_ig = stats.geninvgauss.cdf(x, p=-0.5, b=1 / mu, scale=mu)
    assert_allclose(cdf_ig, stats.invgauss(mu).cdf(x))
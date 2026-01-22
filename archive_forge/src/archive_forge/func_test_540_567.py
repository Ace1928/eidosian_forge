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
def test_540_567():
    assert_almost_equal(stats.norm.cdf(-1.7624320982), 0.03899815971089126, decimal=10, err_msg='test_540_567')
    assert_almost_equal(stats.norm.cdf(-1.7624320983), 0.038998159702449846, decimal=10, err_msg='test_540_567')
    assert_almost_equal(stats.norm.cdf(1.38629436112, loc=0.950273420309, scale=0.204423758009), 0.9835346400430932, decimal=10, err_msg='test_540_567')
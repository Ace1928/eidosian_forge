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
def test_cdf_sf_small_values(self):
    cdfvals = [[-8, 1, 3.8700350466643927e-31], [-4, 2, 8.12983991888114e-21], [-2, 5, 1.5532682678710626e-26], [-9, -1, 2.257176811907681e-19], [-10, -4, 1.523970604832105e-23]]
    for x, a, cdfval in cdfvals:
        p = stats.skewnorm.cdf(x, a)
        assert_allclose(p, cdfval, rtol=1e-08)
        p = stats.skewnorm.sf(-x, -a)
        assert_allclose(p, cdfval, rtol=1e-08)
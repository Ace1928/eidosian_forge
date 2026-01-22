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
def get_moms(lam, sig, mu):
    opK2 = 1.0 + 1 / (lam * sig) ** 2
    exp_skew = 2 / (lam * sig) ** 3 * opK2 ** (-1.5)
    exp_kurt = 6.0 * (1 + (lam * sig) ** 2) ** (-2)
    return [mu + 1 / lam, sig * sig + 1.0 / (lam * lam), exp_skew, exp_kurt]
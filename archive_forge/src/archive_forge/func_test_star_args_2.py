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
def test_star_args_2(self):

    class _dist_gen(stats.rv_continuous):

        def _pdf(self, x, offset, *args):
            extra_kwarg = args[0]
            return stats.norm._pdf(x) * extra_kwarg + offset
    dist = _dist_gen(shapes='offset, extra_kwarg')
    assert_equal(dist.pdf(0.5, offset=111, extra_kwarg=33), stats.norm.pdf(0.5) * 33 + 111)
    assert_equal(dist.pdf(0.5, 111, 33), stats.norm.pdf(0.5) * 33 + 111)
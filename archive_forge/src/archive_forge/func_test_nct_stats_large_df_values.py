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
def test_nct_stats_large_df_values(self):
    nct_mean_df_1000 = stats.nct.mean(1000, 2)
    nct_stats_df_1000 = stats.nct.stats(1000, 2)
    expected_stats_df_1000 = [2.0015015641422464, 1.0040115288163005]
    assert_allclose(nct_mean_df_1000, expected_stats_df_1000[0], rtol=1e-10)
    assert_allclose(nct_stats_df_1000, expected_stats_df_1000, rtol=1e-10)
    nct_mean = stats.nct.mean(100000, 2)
    nct_stats = stats.nct.stats(100000, 2)
    expected_stats = [2.0000150001562518, 1.0000400011500288]
    assert_allclose(nct_mean, expected_stats[0], rtol=1e-10)
    assert_allclose(nct_stats, expected_stats, rtol=1e-09)
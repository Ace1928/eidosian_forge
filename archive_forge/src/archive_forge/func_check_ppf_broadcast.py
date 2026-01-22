import sys
import numpy as np
import numpy.testing as npt
import pytest
from pytest import raises as assert_raises
from scipy.integrate import IntegrationWarning
import itertools
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distcont
from scipy.stats._distn_infrastructure import rv_continuous_frozen
def check_ppf_broadcast(distfn, arg, msg):
    num_repeats = 5
    args = [] * num_repeats
    if arg:
        args = [np.array([_] * num_repeats) for _ in arg]
    median = distfn.ppf(0.5, *arg)
    medians = distfn.ppf(0.5, *args)
    msg += ' - ppf multiple'
    npt.assert_almost_equal(medians, [median] * num_repeats, decimal=7, err_msg=msg)
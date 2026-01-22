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
def check_sample_meanvar_(popmean, popvar, sample):
    if np.isfinite(popmean):
        check_sample_mean(sample, popmean)
    if np.isfinite(popvar):
        check_sample_var(sample, popvar)
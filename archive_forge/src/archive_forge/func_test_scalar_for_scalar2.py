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
def test_scalar_for_scalar2():
    res = stats.norm.fit([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    res = stats.norm.fit_loc_scale([1, 2, 3])
    assert isinstance(res[0], np.number)
    assert isinstance(res[1], np.number)
    res = stats.norm.nnlf((0, 1), [1, 2, 3])
    assert isinstance(res, np.number)
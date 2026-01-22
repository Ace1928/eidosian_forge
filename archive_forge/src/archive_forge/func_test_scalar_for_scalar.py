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
@pytest.mark.parametrize('case', scalar_out + scalars_out)
def test_scalar_for_scalar(case):
    method_name, args = case
    method = getattr(stats.norm(), method_name)
    res = method(*args)
    if case in scalar_out:
        assert isinstance(res, np.number)
    else:
        assert isinstance(res[0], np.number)
        assert isinstance(res[1], np.number)
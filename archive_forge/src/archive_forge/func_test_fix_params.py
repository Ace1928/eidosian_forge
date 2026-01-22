import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_fix_params():
    mod = mlemodel.MLEModel([], 1)
    mod._param_names = ['a', 'b', 'c']
    with mod.fix_params({'b': 1.0}):
        assert_(mod._has_fixed_params)
        assert_equal(mod._fixed_params, {'b': 1.0})
        assert_equal(mod._fixed_params_index, [1])
        assert_equal(mod._free_params_index, [0, 2])
    assert_(not mod._has_fixed_params)
    assert_equal(mod._fixed_params, {})
    assert_equal(mod._fixed_params_index, None)
    assert_equal(mod._free_params_index, None)
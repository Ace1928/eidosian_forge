import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_structural_validate():
    endog = macrodata['infl']
    mod1 = structural.UnobservedComponents(endog, 'rwalk', autoregressive=2)
    assert_raises(ValueError, mod1.fit_constrained, {'AR.L1': 0.5})
    with pytest.raises(ValueError):
        with mod1.fix_params({'ar.L1': 0.5}):
            pass
    assert_raises(ValueError, mod1.fit_constrained, {'ar.L1': 0.5})
    with mod1.fix_params({'ar.L1': 0.5, 'ar.L2': 0.2}):
        assert_(mod1._has_fixed_params)
        assert_equal(mod1._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
        assert_equal(mod1._fixed_params_index, [2, 3])
        assert_equal(mod1._free_params_index, [0, 1])
    res = mod1.fit_constrained({'ar.L1': 0.5, 'ar.L2': 0.2}, start_params=[7.0], disp=False)
    assert_(res._has_fixed_params)
    assert_equal(res._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
    assert_equal(res._fixed_params_index, [2, 3])
    assert_equal(res._free_params_index, [0, 1])
    with mod1.fix_params({'ar.L1': 0.5, 'ar.L2': 0.0}):
        with mod1.fix_params({'ar.L2': 0.2}):
            assert_(mod1._has_fixed_params)
            assert_equal(mod1._fixed_params, {'ar.L1': 0.5, 'ar.L2': 0.2})
            assert_equal(mod1._fixed_params_index, [2, 3])
            assert_equal(mod1._free_params_index, [0, 1])
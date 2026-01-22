import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.simulation_smoother import (
from numpy.testing import assert_equal
def test_filter_methods(self):
    model = self.model
    model.filter_method = 0
    model.filter_conventional = True
    assert_equal(model.filter_method, FILTER_CONVENTIONAL)
    model.filter_collapsed = True
    assert_equal(model.filter_method, FILTER_CONVENTIONAL | FILTER_COLLAPSED)
    model.filter_conventional = False
    assert_equal(model.filter_method, FILTER_COLLAPSED)
    model.set_filter_method(FILTER_AUGMENTED)
    assert_equal(model.filter_method, FILTER_AUGMENTED)
    model.set_filter_method(filter_conventional=True, filter_augmented=False)
    assert_equal(model.filter_method, FILTER_CONVENTIONAL)
    model.filter_method = 0
    for name in model.filter_methods:
        setattr(model, name, True)
    assert_equal(model.filter_method, FILTER_CONVENTIONAL | FILTER_EXACT_INITIAL | FILTER_AUGMENTED | FILTER_SQUARE_ROOT | FILTER_UNIVARIATE | FILTER_COLLAPSED | FILTER_EXTENDED | FILTER_UNSCENTED | FILTER_CONCENTRATED | FILTER_CHANDRASEKHAR)
    for name in model.filter_methods:
        setattr(model, name, False)
    assert_equal(model.filter_method, 0)
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
def test__check_reg_targets():
    EXAMPLES = [('continuous', [1, 2, 3], 1), ('continuous', [[1], [2], [3]], 1), ('continuous-multioutput', [[1, 1], [2, 2], [3, 1]], 2), ('continuous-multioutput', [[5, 1], [4, 2], [3, 1]], 2), ('continuous-multioutput', [[1, 3, 4], [2, 2, 2], [3, 1, 1]], 3)]
    for (type1, y1, n_out1), (type2, y2, n_out2) in product(EXAMPLES, repeat=2):
        if type1 == type2 and n_out1 == n_out2:
            y_type, y_check1, y_check2, multioutput = _check_reg_targets(y1, y2, None)
            assert type1 == y_type
            if type1 == 'continuous':
                assert_array_equal(y_check1, np.reshape(y1, (-1, 1)))
                assert_array_equal(y_check2, np.reshape(y2, (-1, 1)))
            else:
                assert_array_equal(y_check1, y1)
                assert_array_equal(y_check2, y2)
        else:
            with pytest.raises(ValueError):
                _check_reg_targets(y1, y2, None)
import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def test_specific_input_validation(self):
    _rm = _range_matrix
    with pytest.raises(ValueError, match='`partial_guess` can have only as many entries as'):
        quadratic_assignment(np.identity(3), np.identity(3), method=self.method, options={'partial_guess': _rm(5, 2)})
    with pytest.raises(ValueError, match='`partial_guess` must have two columns'):
        quadratic_assignment(np.identity(3), np.identity(3), method=self.method, options={'partial_guess': _range_matrix(2, 3)})
    with pytest.raises(ValueError, match='`partial_guess` must have exactly two'):
        quadratic_assignment(np.identity(3), np.identity(3), method=self.method, options={'partial_guess': np.random.rand(3, 2, 2)})
    with pytest.raises(ValueError, match='`partial_guess` must contain only pos'):
        quadratic_assignment(np.identity(3), np.identity(3), method=self.method, options={'partial_guess': -1 * _range_matrix(2, 2)})
    with pytest.raises(ValueError, match='`partial_guess` entries must be less than number'):
        quadratic_assignment(np.identity(5), np.identity(5), method=self.method, options={'partial_guess': 2 * _range_matrix(4, 2)})
    with pytest.raises(ValueError, match='`partial_guess` column entries must be unique'):
        quadratic_assignment(np.identity(3), np.identity(3), method=self.method, options={'partial_guess': np.ones((2, 2))})
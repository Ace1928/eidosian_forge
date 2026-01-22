import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
def test_cossin_error_non_square():
    with pytest.raises(ValueError, match='only supports square'):
        cossin(np.array([[1, 2]]), 1, 1)
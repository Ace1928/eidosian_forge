import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
def test_cossin_error_missing_partitioning():
    with pytest.raises(ValueError, match='.*exactly four arrays.* got 2'):
        cossin(unitary_group.rvs(2))
    with pytest.raises(ValueError, match='.*might be due to missing p, q'):
        cossin(unitary_group.rvs(4))
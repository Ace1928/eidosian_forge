import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
def test_cossin_error_empty_subblocks():
    with pytest.raises(ValueError, match='x11.*empty'):
        cossin(([], [], [], []))
    with pytest.raises(ValueError, match='x12.*empty'):
        cossin(([1, 2], [], [6, 7], [8, 9, 10]))
    with pytest.raises(ValueError, match='x21.*empty'):
        cossin(([1, 2], [3, 4, 5], [], [8, 9, 10]))
    with pytest.raises(ValueError, match='x22.*empty'):
        cossin(([1, 2], [3, 4, 5], [2], []))
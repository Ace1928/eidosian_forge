import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_power_operator(A):
    assert isinstance(A ** 2, scipy.sparse.sparray), 'Expected array, got matrix'
    npt.assert_equal((A ** 2).todense(), A.todense() ** 2)
    with pytest.raises(NotImplementedError, match='zero power'):
        A ** 0
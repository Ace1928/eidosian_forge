import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_elementwise_rmul(A):
    with pytest.raises(TypeError):
        None * A
    with pytest.raises(ValueError):
        np.eye(3) * scipy.sparse.csr_array(np.arange(6).reshape(2, 3))
    assert np.all(2 * A == A.todense() * 2)
    assert np.all(A.todense() * A == A.todense() ** 2)
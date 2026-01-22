import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_sum(A):
    assert not isinstance(A.sum(axis=0), np.matrix), 'Expected array, got matrix'
    assert A.sum(axis=0).shape == (4,)
    assert A.sum(axis=1).shape == (3,)
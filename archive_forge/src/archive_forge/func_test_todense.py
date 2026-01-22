import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_todense(A):
    assert not isinstance(A.todense(), np.matrix), 'Expected array, got matrix'
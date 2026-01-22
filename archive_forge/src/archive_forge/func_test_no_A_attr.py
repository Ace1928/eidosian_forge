import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_no_A_attr(A):
    with pytest.warns(VisibleDeprecationWarning):
        A.A
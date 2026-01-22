import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_sparrays
def test_getrow_getcol(A):
    assert isinstance(A._getcol(0), scipy.sparse.sparray)
    assert isinstance(A._getrow(0), scipy.sparse.sparray)
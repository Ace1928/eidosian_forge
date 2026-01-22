import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_square_sparrays
def test_onenormest(B):
    C = spla.onenormest(B)
    npt.assert_allclose(C, np.linalg.norm(B.todense(), 1))
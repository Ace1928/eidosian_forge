import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
from numpy.testing import assert_allclose, assert_equal
@pytest.fixture(params=scipy_sparse_classes)
def sp_sparse_cls(request):
    return request.param
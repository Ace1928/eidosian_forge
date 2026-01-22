import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
@pytest.mark.parametrize('fmt', ['csr', 'csc', 'coo', 'lil', 'dok', 'dia', 'bsr'])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('copy', [True, False])
def test_sparse_formats(fmt, normed, copy):
    mat = sparse.diags([1, 1], [-1, 1], shape=(4, 4), format=fmt)
    _check_symmetric_graph_laplacian(mat, normed, copy)
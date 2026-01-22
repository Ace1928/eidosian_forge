import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def test_symmetric_graph_laplacian():
    symmetric_mats = ('np.arange(10) * np.arange(10)[:, np.newaxis]', 'np.ones((7, 7))', 'np.eye(19)', 'sparse.diags([1, 1], [-1, 1], shape=(4, 4))', 'sparse.diags([1, 1], [-1, 1], shape=(4, 4)).toarray()', 'sparse.diags([1, 1], [-1, 1], shape=(4, 4)).todense()', 'np.vander(np.arange(4)) + np.vander(np.arange(4)).T')
    for mat in symmetric_mats:
        for normed in (True, False):
            for copy in (True, False):
                _check_symmetric_graph_laplacian(mat, normed, copy)
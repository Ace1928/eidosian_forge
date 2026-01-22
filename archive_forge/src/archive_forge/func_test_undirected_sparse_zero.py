from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_undirected_sparse_zero():

    def check(method, directed_in):
        if directed_in:
            SP1 = shortest_path(directed_sparse_zero_G, method=method, directed=False, overwrite=False)
            assert_array_almost_equal(SP1, undirected_sparse_zero_SP)
        else:
            SP2 = shortest_path(undirected_sparse_zero_G, method=method, directed=True, overwrite=False)
            assert_array_almost_equal(SP2, undirected_sparse_zero_SP)
    for method in methods:
        for directed_in in (True, False):
            check(method, directed_in)
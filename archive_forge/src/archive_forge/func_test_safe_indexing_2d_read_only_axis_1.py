import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('array_read_only', [True, False])
@pytest.mark.parametrize('indices_read_only', [True, False])
@pytest.mark.parametrize('array_type', ['array', 'sparse', 'dataframe'])
@pytest.mark.parametrize('indices_type', ['array', 'series'])
@pytest.mark.parametrize('axis, expected_array', [(0, [[4, 5, 6], [7, 8, 9]]), (1, [[2, 3], [5, 6], [8, 9]])])
def test_safe_indexing_2d_read_only_axis_1(array_read_only, indices_read_only, array_type, indices_type, axis, expected_array):
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    if array_read_only:
        array.setflags(write=False)
    array = _convert_container(array, array_type)
    indices = np.array([1, 2])
    if indices_read_only:
        indices.setflags(write=False)
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=axis)
    assert_allclose_dense_sparse(subset, _convert_container(expected_array, array_type))
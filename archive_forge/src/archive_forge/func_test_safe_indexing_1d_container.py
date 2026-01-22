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
@pytest.mark.parametrize('array_type', ['list', 'array', 'series'])
@pytest.mark.parametrize('indices_type', ['list', 'tuple', 'array', 'series', 'slice'])
def test_safe_indexing_1d_container(array_type, indices_type):
    indices = [1, 2]
    if indices_type == 'slice' and isinstance(indices[1], int):
        indices[1] += 1
    array = _convert_container([1, 2, 3, 4, 5, 6, 7, 8, 9], array_type)
    indices = _convert_container(indices, indices_type)
    subset = _safe_indexing(array, indices, axis=0)
    assert_allclose_dense_sparse(subset, _convert_container([2, 3], array_type))
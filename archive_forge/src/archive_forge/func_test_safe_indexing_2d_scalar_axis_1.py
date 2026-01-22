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
@pytest.mark.parametrize('array_type, expected_output_type', [('array', 'array'), ('sparse', 'sparse'), ('dataframe', 'series')])
@pytest.mark.parametrize('indices', [2, 'col_2'])
def test_safe_indexing_2d_scalar_axis_1(array_type, expected_output_type, indices):
    columns_name = ['col_0', 'col_1', 'col_2']
    array = _convert_container([[1, 2, 3], [4, 5, 6], [7, 8, 9]], array_type, columns_name)
    if isinstance(indices, str) and array_type != 'dataframe':
        err_msg = 'Specifying the columns using strings is only supported for dataframes'
        with pytest.raises(ValueError, match=err_msg):
            _safe_indexing(array, indices, axis=1)
    else:
        subset = _safe_indexing(array, indices, axis=1)
        expected_output = [3, 6, 9]
        if expected_output_type == 'sparse':
            expected_output = [[3], [6], [9]]
        expected_array = _convert_container(expected_output, expected_output_type)
        assert_allclose_dense_sparse(subset, expected_array)
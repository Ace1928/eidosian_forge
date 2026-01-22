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
def test_column_or_1d():
    EXAMPLES = [('binary', ['spam', 'egg', 'spam']), ('binary', [0, 1, 0, 1]), ('continuous', np.arange(10) / 20.0), ('multiclass', [1, 2, 3]), ('multiclass', [0, 1, 2, 2, 0]), ('multiclass', [[1], [2], [3]]), ('multilabel-indicator', [[0, 1, 0], [0, 0, 1]]), ('multiclass-multioutput', [[1, 2, 3]]), ('multiclass-multioutput', [[1, 1], [2, 2], [3, 1]]), ('multiclass-multioutput', [[5, 1], [4, 2], [3, 1]]), ('multiclass-multioutput', [[1, 2, 3]]), ('continuous-multioutput', np.arange(30).reshape((-1, 3)))]
    for y_type, y in EXAMPLES:
        if y_type in ['binary', 'multiclass', 'continuous']:
            assert_array_equal(column_or_1d(y), np.ravel(y))
        else:
            with pytest.raises(ValueError):
                column_or_1d(y)
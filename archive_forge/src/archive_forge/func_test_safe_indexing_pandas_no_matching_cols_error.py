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
def test_safe_indexing_pandas_no_matching_cols_error():
    pd = pytest.importorskip('pandas')
    err_msg = 'No valid specification of the columns.'
    X = pd.DataFrame(X_toy)
    with pytest.raises(ValueError, match=err_msg):
        _safe_indexing(X, [1.0], axis=1)
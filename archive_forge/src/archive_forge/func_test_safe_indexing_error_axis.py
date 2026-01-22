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
@pytest.mark.parametrize('axis', [None, 3])
def test_safe_indexing_error_axis(axis):
    with pytest.raises(ValueError, match="'axis' should be either 0"):
        _safe_indexing(X_toy, [0, 1], axis=axis)
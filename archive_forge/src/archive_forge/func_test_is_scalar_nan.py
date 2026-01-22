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
@pytest.mark.parametrize('value, result', [(float('nan'), True), (np.nan, True), (float(np.nan), True), (np.float32(np.nan), True), (np.float64(np.nan), True), (0, False), (0.0, False), (None, False), ('', False), ('nan', False), ([np.nan], False), (9867966753463435747313673, False)])
def test_is_scalar_nan(value, result):
    assert is_scalar_nan(value) is result
    assert isinstance(is_scalar_nan(value), bool)
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_array_eq_numpy2d():
    ra = RaggedArray([[1, 2], [1], [1, 2], None, [33, 44]], dtype='int32')
    npa = np.array([[1, 2], [2, 3], [1, 2], [0, 1], [11, 22]], dtype='int32')
    result = ra == npa
    expected = np.array([1, 0, 1, 0, 0], dtype='bool')
    np.testing.assert_array_equal(result, expected)
    result_negated = ra != npa
    expected_negated = ~expected
    np.testing.assert_array_equal(result_negated, expected_negated)
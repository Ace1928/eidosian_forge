from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_construct_ragged_array_fastpath():
    start_indices = np.array([0, 2, 5, 6, 6, 11], dtype='uint16')
    flat_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float32')
    rarray = RaggedArray(dict(start_indices=start_indices, flat_array=flat_array))
    assert np.array_equal(rarray.start_indices, start_indices)
    assert np.array_equal(rarray.flat_array, flat_array)
    object_array = np.asarray(rarray, dtype=object)
    expected_lists = [[0, 1], [2, 3, 4], [5], [], [6, 7, 8, 9, 10], []]
    expected_array = np.array([np.array(v, dtype='float32') for v in expected_lists], dtype='object')
    assert len(object_array) == len(expected_array)
    for a1, a2 in zip(object_array, expected_array):
        np.testing.assert_array_equal(a1, a2)
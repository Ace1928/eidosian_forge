from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
@pytest.mark.parametrize('arg,expected', [([np.array([1, 2], dtype='int64')], 'int64'), ([[True], [False, True]], 'bool'), (np.array([np.array([1, 2], dtype='int8'), np.array([1, 2], dtype='int32')]), 'int32'), ([[3.2], [2]], 'float64'), ([np.array([3.2], dtype='float16'), np.array([2], dtype='float32')], 'float32')])
def test_flat_array_type_inference(arg, expected):
    rarray = RaggedArray(arg)
    assert rarray.flat_array.dtype == np.dtype(expected)
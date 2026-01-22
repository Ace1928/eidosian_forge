from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
@pytest.mark.parametrize('mask', [[1, 1, 1, 1, 1], [0, 1, 0, 1, 1], [0, 0, 0, 0, 0]])
def test_get_item_mask(mask):
    arg = np.array([[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]], dtype=object)
    rarray = RaggedArray(arg, dtype='int16')
    mask = np.array(mask, dtype='bool')
    assert_ragged_arrays_equal(rarray[mask], RaggedArray(arg[mask], dtype='int16'))
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_get_item_slice():
    arg = [[1, 2], [], [10, 20, 30], None, [11, 22, 33, 44]]
    rarray = RaggedArray(arg, dtype='int16')
    assert_ragged_arrays_equal(rarray[:], rarray)
    assert_ragged_arrays_equal(rarray[1:], RaggedArray(arg[1:], dtype='int16'))
    assert_ragged_arrays_equal(rarray[:-1], RaggedArray(arg[:-1], dtype='int16'))
    assert_ragged_arrays_equal(rarray[2:-1], RaggedArray(arg[2:-1], dtype='int16'))
    assert_ragged_arrays_equal(rarray[2:1], RaggedArray(arg[2:1], dtype='int16'))
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs, rename_index, rename_columns', [({'mapper': None, 'axis': 0}, True, False), ({'mapper': None, 'axis': 1}, False, True), ({'index': None}, True, False), ({'columns': None}, False, True), ({'index': None, 'columns': None}, True, True), ({}, False, False)])
def test_rename_axis_none(self, kwargs, rename_index, rename_columns):
    index = Index(list('abc'), name='foo')
    columns = Index(['col1', 'col2'], name='bar')
    data = np.arange(6).reshape(3, 2)
    df = DataFrame(data, index, columns)
    result = df.rename_axis(**kwargs)
    expected_index = index.rename(None) if rename_index else index
    expected_columns = columns.rename(None) if rename_columns else columns
    expected = DataFrame(data, expected_index, expected_columns)
    tm.assert_frame_equal(result, expected)
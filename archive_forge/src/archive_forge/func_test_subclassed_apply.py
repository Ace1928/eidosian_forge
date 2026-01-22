import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclassed_apply(self):

    def check_row_subclass(row):
        assert isinstance(row, tm.SubclassedSeries)

    def stretch(row):
        if row['variable'] == 'height':
            row['value'] += 0.5
        return row
    df = tm.SubclassedDataFrame([['John', 'Doe', 'height', 5.5], ['Mary', 'Bo', 'height', 6.0], ['John', 'Doe', 'weight', 130], ['Mary', 'Bo', 'weight', 150]], columns=['first', 'last', 'variable', 'value'])
    df.apply(lambda x: check_row_subclass(x))
    df.apply(lambda x: check_row_subclass(x), axis=1)
    expected = tm.SubclassedDataFrame([['John', 'Doe', 'height', 6.0], ['Mary', 'Bo', 'height', 6.5], ['John', 'Doe', 'weight', 130], ['Mary', 'Bo', 'weight', 150]], columns=['first', 'last', 'variable', 'value'])
    result = df.apply(lambda x: stretch(x), axis=1)
    assert isinstance(result, tm.SubclassedDataFrame)
    tm.assert_frame_equal(result, expected)
    expected = tm.SubclassedDataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result = df.apply(lambda x: tm.SubclassedSeries([1, 2, 3]), axis=1)
    assert isinstance(result, tm.SubclassedDataFrame)
    tm.assert_frame_equal(result, expected)
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type='expand')
    assert isinstance(result, tm.SubclassedDataFrame)
    tm.assert_frame_equal(result, expected)
    expected = tm.SubclassedSeries([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result = df.apply(lambda x: [1, 2, 3], axis=1)
    assert not isinstance(result, tm.SubclassedDataFrame)
    tm.assert_series_equal(result, expected)
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_axis_parameter(self):
    df1 = DataFrame({'A': [0.1, 0.2]}, index=range(2))
    df2 = DataFrame({'A': [0.3, 0.4]}, index=range(2))
    expected_index = DataFrame({'A': [0.1, 0.2, 0.3, 0.4]}, index=[0, 1, 0, 1])
    concatted_index = concat([df1, df2], axis='index')
    tm.assert_frame_equal(concatted_index, expected_index)
    concatted_row = concat([df1, df2], axis='rows')
    tm.assert_frame_equal(concatted_row, expected_index)
    concatted_0 = concat([df1, df2], axis=0)
    tm.assert_frame_equal(concatted_0, expected_index)
    expected_columns = DataFrame([[0.1, 0.3], [0.2, 0.4]], index=[0, 1], columns=['A', 'A'])
    concatted_columns = concat([df1, df2], axis='columns')
    tm.assert_frame_equal(concatted_columns, expected_columns)
    concatted_1 = concat([df1, df2], axis=1)
    tm.assert_frame_equal(concatted_1, expected_columns)
    series1 = Series([0.1, 0.2])
    series2 = Series([0.3, 0.4])
    expected_index_series = Series([0.1, 0.2, 0.3, 0.4], index=[0, 1, 0, 1])
    concatted_index_series = concat([series1, series2], axis='index')
    tm.assert_series_equal(concatted_index_series, expected_index_series)
    concatted_row_series = concat([series1, series2], axis='rows')
    tm.assert_series_equal(concatted_row_series, expected_index_series)
    concatted_0_series = concat([series1, series2], axis=0)
    tm.assert_series_equal(concatted_0_series, expected_index_series)
    expected_columns_series = DataFrame([[0.1, 0.3], [0.2, 0.4]], index=[0, 1], columns=[0, 1])
    concatted_columns_series = concat([series1, series2], axis='columns')
    tm.assert_frame_equal(concatted_columns_series, expected_columns_series)
    concatted_1_series = concat([series1, series2], axis=1)
    tm.assert_frame_equal(concatted_1_series, expected_columns_series)
    with pytest.raises(ValueError, match='No axis named'):
        concat([series1, series2], axis='something')
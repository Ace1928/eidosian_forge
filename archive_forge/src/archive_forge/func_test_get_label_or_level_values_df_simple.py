import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_get_label_or_level_values_df_simple(df_levels, axis):
    expected_labels, expected_levels = get_labels_levels(df_levels)
    axis = df_levels._get_axis_number(axis)
    if axis == 1:
        df_levels = df_levels.T
    assert_label_values(df_levels, expected_labels, axis=axis)
    assert_level_values(df_levels, expected_levels, axis=axis)
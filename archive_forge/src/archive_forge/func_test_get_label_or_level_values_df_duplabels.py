import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_get_label_or_level_values_df_duplabels(df_duplabels, axis):
    axis = df_duplabels._get_axis_number(axis)
    if axis == 1:
        df_duplabels = df_duplabels.T
    assert_level_values(df_duplabels, ['L1'], axis=axis)
    assert_label_values(df_duplabels, ['L3'], axis=axis)
    if axis == 0:
        expected_msg = "The column label 'L2' is not unique"
    else:
        expected_msg = "The index label 'L2' is not unique"
    with pytest.raises(ValueError, match=expected_msg):
        assert_label_values(df_duplabels, ['L2'], axis=axis)
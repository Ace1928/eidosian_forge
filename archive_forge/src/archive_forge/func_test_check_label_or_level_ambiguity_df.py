import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_check_label_or_level_ambiguity_df(df_ambig, axis):
    axis = df_ambig._get_axis_number(axis)
    if axis == 1:
        df_ambig = df_ambig.T
        msg = "'L1' is both a column level and an index label"
    else:
        msg = "'L1' is both an index level and a column label"
    with pytest.raises(ValueError, match=msg):
        df_ambig._check_label_or_level_ambiguity('L1', axis=axis)
    df_ambig._check_label_or_level_ambiguity('L2', axis=axis)
    assert not df_ambig._check_label_or_level_ambiguity('L3', axis=axis)
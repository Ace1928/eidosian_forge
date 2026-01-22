import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_get_label_or_level_values_series_axis0(df):
    s = df.set_index('L1').L2
    assert_level_values(s, ['L1'], axis=0)
    s = df.set_index(['L1', 'L2']).L3
    assert_level_values(s, ['L1', 'L2'], axis=0)
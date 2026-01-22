import pytest
from pandas.core.dtypes.missing import array_equivalent
import pandas as pd
def test_is_level_reference_series_simple_axis0(df):
    s = df.set_index('L1').L2
    assert_level_reference(s, ['L1'], axis=0)
    assert not s._is_level_reference('L2')
    s = df.set_index(['L1', 'L2']).L3
    assert_level_reference(s, ['L1', 'L2'], axis=0)
    assert not s._is_level_reference('L3')
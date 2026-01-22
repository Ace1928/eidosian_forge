import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_string_matching_single_index(df):
    for df_swap in [df.swaplevel(), df.swaplevel(0), df.swaplevel(0, 1)]:
        df_swap = df_swap.sort_index()
        just_a = df_swap.loc['a']
        result = just_a.loc['2016-01-01']
        expected = df.loc[IndexSlice[:, 'a'], :].iloc[0:2]
        expected.index = expected.index.droplevel(1)
        tm.assert_frame_equal(result, expected)
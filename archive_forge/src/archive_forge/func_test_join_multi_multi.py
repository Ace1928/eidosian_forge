import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_join_multi_multi(self, left_multi, right_multi, join_type, on_cols_multi):
    left_names = left_multi.index.names
    right_names = right_multi.index.names
    if join_type == 'right':
        level_order = right_names + left_names.difference(right_names)
    else:
        level_order = left_names + right_names.difference(left_names)
    expected = merge(left_multi.reset_index(), right_multi.reset_index(), how=join_type, on=on_cols_multi).set_index(level_order).sort_index()
    result = left_multi.join(right_multi, how=join_type).sort_index()
    tm.assert_frame_equal(result, expected)
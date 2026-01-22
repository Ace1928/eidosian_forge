import builtins
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bool_agg_func', ['any', 'all'])
def test_bool_aggs_dup_column_labels(bool_agg_func):
    df = DataFrame([[True, True]], columns=['a', 'a'])
    grp_by = df.groupby([0])
    result = getattr(grp_by, bool_agg_func)()
    expected = df.set_axis(np.array([0]))
    tm.assert_frame_equal(result, expected)
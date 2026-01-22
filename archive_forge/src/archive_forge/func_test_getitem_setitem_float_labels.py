from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_setitem_float_labels(self, using_array_manager):
    index = Index([1.5, 2, 3, 4, 5])
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)
    result = df.loc[1.5:4]
    expected = df.reindex([1.5, 2, 3, 4])
    tm.assert_frame_equal(result, expected)
    assert len(result) == 4
    result = df.loc[4:5]
    expected = df.reindex([4, 5])
    tm.assert_frame_equal(result, expected, check_index_type=False)
    assert len(result) == 2
    result = df.loc[4:5]
    expected = df.reindex([4.0, 5.0])
    tm.assert_frame_equal(result, expected)
    assert len(result) == 2
    result = df.loc[1:2]
    expected = df.iloc[0:2]
    tm.assert_frame_equal(result, expected)
    expected = df.iloc[0:2]
    msg = 'The behavior of obj\\[i:j\\] with a float-dtype index'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df[1:2]
    tm.assert_frame_equal(result, expected)
    index = Index([1.0, 2.5, 3.5, 4.5, 5.0])
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)), index=index)
    msg = 'cannot do positional indexing on Index with these indexers \\[1.0\\] of type float'
    with pytest.raises(TypeError, match=msg):
        df.iloc[1.0:5]
    result = df.iloc[4:5]
    expected = df.reindex([5.0])
    tm.assert_frame_equal(result, expected)
    assert len(result) == 1
    cp = df.copy()
    with pytest.raises(TypeError, match=_slice_msg):
        cp.iloc[1.0:5] = 0
    with pytest.raises(TypeError, match=msg):
        result = cp.iloc[1.0:5] == 0
    assert result.values.all()
    assert (cp.iloc[0:1] == df.iloc[0:1]).values.all()
    cp = df.copy()
    cp.iloc[4:5] = 0
    assert (cp.iloc[4:5] == 0).values.all()
    assert (cp.iloc[0:4] == df.iloc[0:4]).values.all()
    result = df.loc[1.0:5]
    expected = df
    tm.assert_frame_equal(result, expected)
    assert len(result) == 5
    result = df.loc[1.1:5]
    expected = df.reindex([2.5, 3.5, 4.5, 5.0])
    tm.assert_frame_equal(result, expected)
    assert len(result) == 4
    result = df.loc[4.51:5]
    expected = df.reindex([5.0])
    tm.assert_frame_equal(result, expected)
    assert len(result) == 1
    result = df.loc[1.0:5.0]
    expected = df.reindex([1.0, 2.5, 3.5, 4.5, 5.0])
    tm.assert_frame_equal(result, expected)
    assert len(result) == 5
    cp = df.copy()
    cp.loc[1.0:5.0] = 0
    result = cp.loc[1.0:5.0]
    assert (result == 0).values.all()
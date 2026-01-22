from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_combineFrame(self, float_frame, mixed_float_frame, mixed_int_frame):
    frame_copy = float_frame.reindex(float_frame.index[::2])
    del frame_copy['D']
    frame_copy.loc[:frame_copy.index[4], 'C'] = np.nan
    added = float_frame + frame_copy
    indexer = added['A'].dropna().index
    exp = (float_frame['A'] * 2).copy()
    tm.assert_series_equal(added['A'].dropna(), exp.loc[indexer])
    exp.loc[~exp.index.isin(indexer)] = np.nan
    tm.assert_series_equal(added['A'], exp.loc[added['A'].index])
    assert np.isnan(added['C'].reindex(frame_copy.index)[:5]).all()
    assert np.isnan(added['D']).all()
    self_added = float_frame + float_frame
    tm.assert_index_equal(self_added.index, float_frame.index)
    added_rev = frame_copy + float_frame
    assert np.isnan(added['D']).all()
    assert np.isnan(added_rev['D']).all()
    plus_empty = float_frame + DataFrame()
    assert np.isnan(plus_empty.values).all()
    empty_plus = DataFrame() + float_frame
    assert np.isnan(empty_plus.values).all()
    empty_empty = DataFrame() + DataFrame()
    assert empty_empty.empty
    reverse = float_frame.reindex(columns=float_frame.columns[::-1])
    tm.assert_frame_equal(reverse + float_frame, float_frame * 2)
    added = float_frame + mixed_float_frame
    _check_mixed_float(added, dtype='float64')
    added = mixed_float_frame + float_frame
    _check_mixed_float(added, dtype='float64')
    added = mixed_float_frame + mixed_float_frame
    _check_mixed_float(added, dtype={'C': None})
    added = float_frame + mixed_int_frame
    _check_mixed_float(added, dtype='float64')
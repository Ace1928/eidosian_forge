from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_non_slice_like_blocks(using_array_manager):
    mi = MultiIndex.from_product([range(5), ['A', 'B', 'C']])
    df = DataFrame({0: np.random.default_rng(2).standard_normal(15), 1: np.random.default_rng(2).standard_normal(15).astype(np.int64), 2: np.random.default_rng(2).standard_normal(15), 3: np.random.default_rng(2).standard_normal(15)}, index=mi)
    if not using_array_manager:
        assert any((not x.mgr_locs.is_slice_like for x in df._mgr.blocks))
    res = df.unstack()
    expected = pd.concat([df[n].unstack() for n in range(4)], keys=range(4), axis=1)
    tm.assert_frame_equal(res, expected)
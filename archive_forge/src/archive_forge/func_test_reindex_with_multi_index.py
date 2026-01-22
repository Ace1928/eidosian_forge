from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_with_multi_index(self):
    df = DataFrame({'a': [-1] * 7 + [0] * 7 + [1] * 7, 'b': list(range(7)) * 3, 'c': ['A', 'B', 'C', 'D', 'E', 'F', 'G'] * 3}).set_index(['a', 'b'])
    new_index = [0.5, 2.0, 5.0, 5.8]
    new_multi_index = MultiIndex.from_product([[0], new_index], names=['a', 'b'])
    reindexed = df.reindex(new_multi_index)
    expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': [np.nan, 'C', 'F', np.nan]}).set_index(['a', 'b'])
    tm.assert_frame_equal(expected, reindexed)
    expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': ['B', 'C', 'F', 'G']}).set_index(['a', 'b'])
    reindexed_with_backfilling = df.reindex(new_multi_index, method='bfill')
    tm.assert_frame_equal(expected, reindexed_with_backfilling)
    reindexed_with_backfilling = df.reindex(new_multi_index, method='backfill')
    tm.assert_frame_equal(expected, reindexed_with_backfilling)
    expected = DataFrame({'a': [0] * 4, 'b': new_index, 'c': ['A', 'C', 'F', 'F']}).set_index(['a', 'b'])
    reindexed_with_padding = df.reindex(new_multi_index, method='pad')
    tm.assert_frame_equal(expected, reindexed_with_padding)
    reindexed_with_padding = df.reindex(new_multi_index, method='ffill')
    tm.assert_frame_equal(expected, reindexed_with_padding)
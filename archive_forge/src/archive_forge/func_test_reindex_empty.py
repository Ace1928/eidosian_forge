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
@pytest.mark.parametrize('src_idx', [Index([]), CategoricalIndex([])])
@pytest.mark.parametrize('cat_idx', [Index([]), CategoricalIndex([]), Index(['A', 'B']), CategoricalIndex(['A', 'B']), Index(['A', 'A']), CategoricalIndex(['A', 'A'])])
def test_reindex_empty(self, src_idx, cat_idx):
    df = DataFrame(columns=src_idx, index=['K'], dtype='f8')
    result = df.reindex(columns=cat_idx)
    expected = DataFrame(index=['K'], columns=cat_idx, dtype='f8')
    tm.assert_frame_equal(result, expected)
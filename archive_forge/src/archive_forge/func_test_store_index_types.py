import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('format', ['table', 'fixed'])
@pytest.mark.parametrize('index', [Index([str(i) for i in range(10)]), Index(np.arange(10, dtype=float)), Index(np.arange(10)), date_range('2020-01-01', periods=10), pd.period_range('2020-01-01', periods=10)])
def test_store_index_types(setup_path, format, index):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('AB'), index=index)
        _maybe_remove(store, 'df')
        store.put('df', df, format=format)
        tm.assert_frame_equal(df, store['df'])
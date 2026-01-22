import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import Term
@pytest.mark.parametrize('where', ['', (), (None,), [], [None]])
def test_select_empty_where(tmp_path, where):
    df = DataFrame([1, 2, 3])
    path = tmp_path / 'empty_where.h5'
    with HDFStore(path) as store:
        store.put('df', df, 't')
        result = read_hdf(store, 'df', where=where)
        tm.assert_frame_equal(result, df)
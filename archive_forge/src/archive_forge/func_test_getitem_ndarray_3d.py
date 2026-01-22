import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
@pytest.mark.filterwarnings('ignore:Series.__getitem__ treating keys as positions is deprecated:FutureWarning')
def test_getitem_ndarray_3d(self, index, frame_or_series, indexer_sli, using_array_manager):
    obj = gen_obj(frame_or_series, index)
    idxr = indexer_sli(obj)
    nd3 = np.random.default_rng(2).integers(5, size=(2, 2, 2))
    msgs = []
    if frame_or_series is Series and indexer_sli in [tm.setitem, tm.iloc]:
        msgs.append('Wrong number of dimensions. values.ndim > ndim \\[3 > 1\\]')
        if using_array_manager:
            msgs.append('Passed array should be 1-dimensional')
    if frame_or_series is Series or indexer_sli is tm.iloc:
        msgs.append('Buffer has wrong number of dimensions \\(expected 1, got 3\\)')
        if using_array_manager:
            msgs.append('indexer should be 1-dimensional')
    if indexer_sli is tm.loc or (frame_or_series is Series and indexer_sli is tm.setitem):
        msgs.append('Cannot index with multidimensional key')
    if frame_or_series is DataFrame and indexer_sli is tm.setitem:
        msgs.append('Index data must be 1-dimensional')
    if isinstance(index, pd.IntervalIndex) and indexer_sli is tm.iloc:
        msgs.append('Index data must be 1-dimensional')
    if isinstance(index, (pd.TimedeltaIndex, pd.DatetimeIndex, pd.PeriodIndex)):
        msgs.append('Data must be 1-dimensional')
    if len(index) == 0 or isinstance(index, pd.MultiIndex):
        msgs.append('positional indexers are out-of-bounds')
    if type(index) is Index and (not isinstance(index._values, np.ndarray)):
        msgs.append('values must be a 1D array')
        msgs.append('only handle 1-dimensional arrays')
    msg = '|'.join(msgs)
    potential_errors = (IndexError, ValueError, NotImplementedError)
    with pytest.raises(potential_errors, match=msg):
        idxr[nd3]
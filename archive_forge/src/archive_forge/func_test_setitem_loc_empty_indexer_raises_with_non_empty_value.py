from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@pytest.mark.xfail(reason='Currently empty indexers are treated as all False')
@pytest.mark.parametrize('box', [list, np.array, Series])
def test_setitem_loc_empty_indexer_raises_with_non_empty_value(self, box):
    df = DataFrame({'a': ['a'], 'b': [1], 'c': [1]})
    if box == Series:
        indexer = box([], dtype='object')
    else:
        indexer = box([])
    msg = 'Must have equal len keys and value when setting with an iterable'
    with pytest.raises(ValueError, match=msg):
        df.loc[indexer, ['b']] = [1]
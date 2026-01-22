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
@pytest.mark.parametrize('box', [list, np.array, pd.array, pd.Categorical, Index])
@pytest.mark.parametrize('key', [[0, 1], slice(0, 2), np.array([True, True, False])])
def test_setitem_td64_string_values(self, indexer_sli, key, box):
    if isinstance(key, slice) and indexer_sli is tm.loc:
        key = slice(0, 1)
    tdi = timedelta_range('1 Day', periods=3)
    ser = Series(tdi.copy(deep=True))
    values = ser._values
    newvals = box(['10 Days', '44 hours'])
    values._validate_setitem_value(newvals)
    indexer_sli(ser)[key] = newvals
    assert ser._values._ndarray is values._ndarray
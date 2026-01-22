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
@pytest.mark.parametrize('indexer', [tm.getitem, tm.loc, tm.iloc])
def test_getitem_preserve_object_index_with_dates(self, indexer):
    idx = date_range('2012', periods=3).astype(object)
    df = DataFrame({0: [1, 2, 3]}, index=idx)
    assert df.index.dtype == object
    if indexer is tm.getitem:
        ser = indexer(df)[0]
    else:
        ser = indexer(df)[:, 0]
    assert ser.index.dtype == object
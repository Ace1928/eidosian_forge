from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_setitem_slice_integers(self):
    ser = Series(np.random.default_rng(2).standard_normal(8), index=[2, 4, 6, 8, 10, 12, 14, 16])
    ser[:4] = 0
    assert (ser[:4] == 0).all()
    assert not (ser[4:] == 0).any()
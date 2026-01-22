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
def test_slice_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
    super().test_slice_key(obj, key, expected, warn, val, indexer_sli, is_inplace)
    if isinstance(val, float):
        raise AssertionError('xfail not relevant for this test.')
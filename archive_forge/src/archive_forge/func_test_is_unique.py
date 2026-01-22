from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_is_unique(self, simple_index):
    index = simple_index.drop_duplicates()
    assert index.is_unique is True
    index_empty = index[:0]
    assert index_empty.is_unique is True
    index_dup = index.insert(0, index[0])
    assert index_dup.is_unique is False
    index_na = index.insert(0, np.nan)
    assert index_na.is_unique is True
    index_na_dup = index_na.insert(0, np.nan)
    assert index_na_dup.is_unique is False
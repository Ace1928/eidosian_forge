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
def test_map_str(self, simple_index):
    idx = simple_index
    result = idx.map(str)
    expected = Index([str(x) for x in idx], dtype=object)
    tm.assert_index_equal(result, expected)
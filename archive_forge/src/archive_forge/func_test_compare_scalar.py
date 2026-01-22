from __future__ import annotations
from typing import final
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
def test_compare_scalar(self, data, comparison_op):
    ser = pd.Series(data)
    self._compare_other(ser, data, comparison_op, 0)
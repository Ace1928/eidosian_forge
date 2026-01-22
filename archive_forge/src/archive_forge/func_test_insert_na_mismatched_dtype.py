import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas._libs.arrays import NDArrayBacked
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_insert_na_mismatched_dtype(self):
    ci = CategoricalIndex([0, 1, 1])
    result = ci.insert(0, pd.NaT)
    expected = Index([pd.NaT, 0, 1, 1], dtype=object)
    tm.assert_index_equal(result, expected)
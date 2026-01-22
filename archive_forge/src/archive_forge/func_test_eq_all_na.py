import pickle
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_ import (
from pandas.core.arrays.string_arrow import (
def test_eq_all_na():
    pytest.importorskip('pyarrow')
    a = pd.array([pd.NA, pd.NA], dtype=StringDtype('pyarrow'))
    result = a == a
    expected = pd.array([pd.NA, pd.NA], dtype='boolean[pyarrow]')
    tm.assert_extension_array_equal(result, expected)
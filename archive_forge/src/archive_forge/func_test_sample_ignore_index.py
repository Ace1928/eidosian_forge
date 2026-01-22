import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_ignore_index(self):
    df = DataFrame({'col1': range(10, 20), 'col2': range(20, 30), 'colString': ['a'] * 10})
    result = df.sample(3, ignore_index=True)
    expected_index = Index(range(3))
    tm.assert_index_equal(result.index, expected_index, exact=True)
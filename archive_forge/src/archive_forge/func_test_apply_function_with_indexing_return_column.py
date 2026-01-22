from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_function_with_indexing_return_column():
    df = DataFrame({'foo1': ['one', 'two', 'two', 'three', 'one', 'two'], 'foo2': [1, 2, 4, 4, 5, 6]})
    result = df.groupby('foo1', as_index=False).apply(lambda x: x.mean())
    expected = DataFrame({'foo1': ['one', 'three', 'two'], 'foo2': [3.0, 4.0, 4.0]})
    tm.assert_frame_equal(result, expected)
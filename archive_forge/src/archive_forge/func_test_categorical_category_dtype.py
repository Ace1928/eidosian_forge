from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ordered', [False, True])
@pytest.mark.parametrize('categories', [['a', 'b', 'c'], ['a', 'c', 'b'], ['a', 'b', 'c', 'd'], ['c', 'b', 'a']])
def test_categorical_category_dtype(all_parsers, categories, ordered):
    parser = all_parsers
    data = 'a,b\n1,a\n1,b\n1,b\n2,c'
    expected = DataFrame({'a': [1, 1, 1, 2], 'b': Categorical(['a', 'b', 'b', 'c'], categories=categories, ordered=ordered)})
    dtype = {'b': CategoricalDtype(categories=categories, ordered=ordered)}
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)
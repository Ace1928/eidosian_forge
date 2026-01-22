import numpy as np
import pytest
from pandas.core.dtypes.concat import union_categoricals
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_union_categoricals_sort_false(self):
    c1 = Categorical(['x', 'y', 'z'])
    c2 = Categorical(['a', 'b', 'c'])
    result = union_categoricals([c1, c2], sort_categories=False)
    expected = Categorical(['x', 'y', 'z', 'a', 'b', 'c'], categories=['x', 'y', 'z', 'a', 'b', 'c'])
    tm.assert_categorical_equal(result, expected)
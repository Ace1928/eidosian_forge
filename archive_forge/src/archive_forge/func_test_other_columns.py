from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_other_columns(self, left, right, using_infer_string):
    right = right.assign(Z=right.Z.astype('category'))
    merged = merge(left, right, on='X')
    result = merged.dtypes.sort_index()
    dtype = np.dtype('O') if not using_infer_string else 'string'
    expected = Series([CategoricalDtype(categories=['foo', 'bar']), dtype, CategoricalDtype(categories=[1, 2])], index=['X', 'Y', 'Z'])
    tm.assert_series_equal(result, expected)
    assert left.X.values._categories_match_up_to_permutation(merged.X.values)
    assert right.Z.values._categories_match_up_to_permutation(merged.Z.values)
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
@pytest.mark.parametrize('on,left_on,right_on,left_index,right_index,nm', [(['outer', 'inner'], None, None, False, False, 'B'), (None, None, None, True, True, 'B'), (None, ['outer', 'inner'], None, False, True, 'B'), (None, None, ['outer', 'inner'], True, False, 'B'), (['outer', 'inner'], None, None, False, False, None), (None, None, None, True, True, None), (None, ['outer', 'inner'], None, False, True, None), (None, None, ['outer', 'inner'], True, False, None)])
def test_merge_series(on, left_on, right_on, left_index, right_index, nm):
    a = DataFrame({'A': [1, 2, 3, 4]}, index=MultiIndex.from_product([['a', 'b'], [0, 1]], names=['outer', 'inner']))
    b = Series([1, 2, 3, 4], index=MultiIndex.from_product([['a', 'b'], [1, 2]], names=['outer', 'inner']), name=nm)
    expected = DataFrame({'A': [2, 4], 'B': [1, 3]}, index=MultiIndex.from_product([['a', 'b'], [1]], names=['outer', 'inner']))
    if nm is not None:
        result = merge(a, b, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index)
        tm.assert_frame_equal(result, expected)
    else:
        msg = 'Cannot merge a Series without a name'
        with pytest.raises(ValueError, match=msg):
            result = merge(a, b, on=on, left_on=left_on, right_on=right_on, left_index=left_index, right_index=right_index)
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('codes', ([[0, 0, 1, 1], [0, 1, 0, 1]], [[0, 0, -1, 1], [0, 1, 0, 1]]))
def test_rest_index_multiindex_categorical_with_missing_values(self, codes):
    index = MultiIndex([CategoricalIndex(['A', 'B']), CategoricalIndex(['a', 'b'])], codes)
    data = {'col': range(len(index))}
    df = DataFrame(data=data, index=index)
    expected = DataFrame({'level_0': Categorical.from_codes(codes[0], categories=['A', 'B']), 'level_1': Categorical.from_codes(codes[1], categories=['a', 'b']), 'col': range(4)})
    res = df.reset_index()
    tm.assert_frame_equal(res, expected)
    res = expected.set_index(['level_0', 'level_1']).reset_index()
    tm.assert_frame_equal(res, expected)
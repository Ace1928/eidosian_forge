from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_tuples_with_tuple_label():
    expected = pd.DataFrame([[2, 1, 2], [4, (1, 2), 3]], columns=['a', 'b', 'c']).set_index(['a', 'b'])
    idx = MultiIndex.from_tuples([(2, 1), (4, (1, 2))], names=('a', 'b'))
    result = pd.DataFrame([2, 3], columns=['c'], index=idx)
    tm.assert_frame_equal(expected, result)
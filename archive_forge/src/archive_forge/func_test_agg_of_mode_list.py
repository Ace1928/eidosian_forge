import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('test, constant', [([[20, 'A'], [20, 'B'], [10, 'C']], {0: [10, 20], 1: ['C', ['A', 'B']]}), ([[20, 'A'], [20, 'B'], [30, 'C']], {0: [20, 30], 1: [['A', 'B'], 'C']}), ([['a', 1], ['a', 1], ['b', 2], ['b', 3]], {0: ['a', 'b'], 1: [1, [2, 3]]}), pytest.param([['a', 1], ['a', 2], ['b', 3], ['b', 3]], {0: ['a', 'b'], 1: [[1, 2], 3]}, marks=pytest.mark.xfail)])
def test_agg_of_mode_list(test, constant):
    df1 = DataFrame(test)
    result = df1.groupby(0).agg(Series.mode)
    expected = DataFrame(constant)
    expected = expected.set_index(0)
    tm.assert_frame_equal(result, expected)
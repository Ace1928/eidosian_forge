from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.usefixtures('pyarrow_xfail')
def test_ea_int_avoid_overflow(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,1\n,1\n1582218195625938945,1\n'
    result = parser.read_csv(StringIO(data), dtype={'a': 'Int64'})
    expected = DataFrame({'a': IntegerArray(np.array([1, 1, 1582218195625938945]), np.array([False, True, False])), 'b': 1})
    tm.assert_frame_equal(result, expected)
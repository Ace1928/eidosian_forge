from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('dtype', list(np.typecodes['AllInteger'] + np.typecodes['Float']))
def test_numeric_dtype(all_parsers, dtype):
    data = '0\n1'
    parser = all_parsers
    expected = DataFrame([0, 1], dtype=dtype)
    result = parser.read_csv(StringIO(data), header=None, dtype=dtype)
    tm.assert_frame_equal(expected, result)
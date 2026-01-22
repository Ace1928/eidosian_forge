from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_names_longer_than_header_but_equal_with_data_rows(all_parsers):
    parser = all_parsers
    data = 'a, b\n1,2,3\n5,6,4\n'
    result = parser.read_csv(StringIO(data), header=0, names=['A', 'B', 'C'])
    expected = DataFrame({'A': [1, 5], 'B': [2, 6], 'C': [3, 4]})
    tm.assert_frame_equal(result, expected)
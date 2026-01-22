from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_large_difference_in_columns(c_parser_only):
    parser = c_parser_only
    count = 10000
    large_row = ('X,' * count)[:-1] + '\n'
    normal_row = 'XXXXXX XXXXXX,111111111111111\n'
    test_input = (large_row + normal_row * 6)[:-1]
    result = parser.read_csv(StringIO(test_input), header=None, usecols=[0])
    rows = test_input.split('\n')
    expected = DataFrame([row.split(',')[0] for row in rows])
    tm.assert_frame_equal(result, expected)
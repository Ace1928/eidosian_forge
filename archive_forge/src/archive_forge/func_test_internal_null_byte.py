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
def test_internal_null_byte(c_parser_only):
    parser = c_parser_only
    names = ['a', 'b', 'c']
    data = '1,2,3\n4,\x00,6\n7,8,9'
    expected = DataFrame([[1, 2.0, 3], [4, np.nan, 6], [7, 8, 9]], columns=names)
    result = parser.read_csv(StringIO(data), names=names)
    tm.assert_frame_equal(result, expected)
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
def test_unix_style_breaks(c_parser_only):
    parser = c_parser_only
    with tm.ensure_clean() as path:
        with open(path, 'w', newline='\n', encoding='utf-8') as f:
            f.write('blah\n\ncol_1,col_2,col_3\n\n')
        result = parser.read_csv(path, skiprows=2, encoding='utf-8', engine='c')
    expected = DataFrame(columns=['col_1', 'col_2', 'col_3'])
    tm.assert_frame_equal(result, expected)
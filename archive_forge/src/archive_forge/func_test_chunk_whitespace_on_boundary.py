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
def test_chunk_whitespace_on_boundary(c_parser_only):
    parser = c_parser_only
    chunk1 = 'a' * (1024 * 256 - 2) + '\na'
    chunk2 = '\n a'
    result = parser.read_csv(StringIO(chunk1 + chunk2), header=None)
    expected = DataFrame(['a' * (1024 * 256 - 2), 'a', ' a'])
    tm.assert_frame_equal(result, expected)
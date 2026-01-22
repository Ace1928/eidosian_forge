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
def test_tokenize_CR_with_quoting(c_parser_only):
    parser = c_parser_only
    data = ' a,b,c\r"a,b","e,d","f,f"'
    result = parser.read_csv(StringIO(data), header=None)
    expected = parser.read_csv(StringIO(data.replace('\r', '\n')), header=None)
    tm.assert_frame_equal(result, expected)
    result = parser.read_csv(StringIO(data))
    expected = parser.read_csv(StringIO(data.replace('\r', '\n')))
    tm.assert_frame_equal(result, expected)
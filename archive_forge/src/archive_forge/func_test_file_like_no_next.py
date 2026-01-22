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
def test_file_like_no_next(c_parser_only):

    class NoNextBuffer(StringIO):

        def __next__(self):
            raise AttributeError('No next method')
        next = __next__
    parser = c_parser_only
    data = 'a\n1'
    expected = DataFrame({'a': [1]})
    result = parser.read_csv(NoNextBuffer(data))
    tm.assert_frame_equal(result, expected)
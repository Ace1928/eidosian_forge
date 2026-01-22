from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_no_header_pyarrow(pyarrow_parser_only):
    parser = pyarrow_parser_only
    data = '\na,i,x\nb,j,y\n'
    result = parser.read_csv(StringIO(data), header=None, usecols=[0, 1], dtype='string[pyarrow]', dtype_backend='pyarrow', engine='pyarrow')
    expected = DataFrame([['a', 'i'], ['b', 'j']], dtype='string[pyarrow]')
    tm.assert_frame_equal(result, expected)
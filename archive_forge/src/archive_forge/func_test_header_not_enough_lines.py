from io import (
import numpy as np
import pytest
import pandas._libs.parsers as parser
from pandas._libs.parsers import TextReader
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import (
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs
def test_header_not_enough_lines(self):
    data = 'skip this\nskip this\na,b,c\n1,2,3\n4,5,6'
    reader = TextReader(StringIO(data), delimiter=',', header=2)
    header = reader.header
    expected = [['a', 'b', 'c']]
    assert header == expected
    recs = reader.read()
    expected = {0: np.array([1, 4], dtype=np.int64), 1: np.array([2, 5], dtype=np.int64), 2: np.array([3, 6], dtype=np.int64)}
    assert_array_dicts_equal(recs, expected)
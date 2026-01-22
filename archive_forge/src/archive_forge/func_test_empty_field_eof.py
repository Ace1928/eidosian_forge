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
def test_empty_field_eof(self):
    data = 'a,b,c\n1,2,3\n4,,'
    result = TextReader(StringIO(data), delimiter=',').read()
    expected = {0: np.array([1, 4], dtype=np.int64), 1: np.array(['2', ''], dtype=object), 2: np.array(['3', ''], dtype=object)}
    assert_array_dicts_equal(result, expected)
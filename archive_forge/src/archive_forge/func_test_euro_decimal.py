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
def test_euro_decimal(self):
    data = '12345,67\n345,678'
    reader = TextReader(StringIO(data), delimiter=':', decimal=',', header=None)
    result = reader.read()
    expected = np.array([12345.67, 345.678])
    tm.assert_almost_equal(result[0], expected)
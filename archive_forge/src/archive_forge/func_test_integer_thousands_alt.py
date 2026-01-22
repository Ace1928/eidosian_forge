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
def test_integer_thousands_alt(self):
    data = '123.456\n12.500'
    reader = TextFileReader(StringIO(data), delimiter=':', thousands='.', header=None)
    result = reader.read()
    expected = DataFrame([123456, 12500])
    tm.assert_frame_equal(result, expected)
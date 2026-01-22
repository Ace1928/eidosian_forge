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
def test_embedded_newline(self):
    data = 'a\n"hello\nthere"\nthis'
    reader = TextReader(StringIO(data), header=None)
    result = reader.read()
    expected = np.array(['a', 'hello\nthere', 'this'], dtype=np.object_)
    tm.assert_numpy_array_equal(result[0], expected)
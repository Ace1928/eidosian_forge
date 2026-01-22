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
def test_delimit_whitespace(self):
    data = 'a  b\na\t\t "b"\n"a"\t \t b'
    reader = TextReader(StringIO(data), delim_whitespace=True, header=None)
    result = reader.read()
    tm.assert_numpy_array_equal(result[0], np.array(['a', 'a', 'a'], dtype=np.object_))
    tm.assert_numpy_array_equal(result[1], np.array(['b', 'b', 'b'], dtype=np.object_))
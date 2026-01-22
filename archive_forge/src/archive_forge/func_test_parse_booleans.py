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
def test_parse_booleans(self):
    data = 'True\nFalse\nTrue\nTrue'
    reader = TextReader(StringIO(data), header=None)
    result = reader.read()
    assert result[0].dtype == np.bool_
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
def test_pass_dtype(self):
    data = 'one,two\n1,a\n2,b\n3,c\n4,d'

    def _make_reader(**kwds):
        if 'dtype' in kwds:
            kwds['dtype'] = ensure_dtype_objs(kwds['dtype'])
        return TextReader(StringIO(data), delimiter=',', **kwds)
    reader = _make_reader(dtype={'one': 'u1', 1: 'S1'})
    result = reader.read()
    assert result[0].dtype == 'u1'
    assert result[1].dtype == 'S1'
    reader = _make_reader(dtype={'one': np.uint8, 1: object})
    result = reader.read()
    assert result[0].dtype == 'u1'
    assert result[1].dtype == 'O'
    reader = _make_reader(dtype={'one': np.dtype('u1'), 1: np.dtype('O')})
    result = reader.read()
    assert result[0].dtype == 'u1'
    assert result[1].dtype == 'O'
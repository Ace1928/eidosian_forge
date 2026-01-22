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
@pytest.mark.parametrize('text, kwargs', [('a,b,c\r1,2,3\r4,5,6\r7,8,9\r10,11,12', {'delimiter': ','}), ('a  b  c\r1  2  3\r4  5  6\r7  8  9\r10  11  12', {'delim_whitespace': True}), ('a,b,c\r1,2,3\r4,5,6\r,88,9\r10,11,12', {'delimiter': ','}), ('A,B,C,D,E,F,G,H,I,J,K,L,M,N,O\rAAAAA,BBBBB,0,0,0,0,0,0,0,0,0,0,0,0,0\r,BBBBB,0,0,0,0,0,0,0,0,0,0,0,0,0', {'delimiter': ','}), ('A  B  C\r  2  3\r4  5  6', {'delim_whitespace': True}), ('A B C\r2 3\r4 5 6', {'delim_whitespace': True})])
def test_cr_delimited(self, text, kwargs):
    nice_text = text.replace('\r', '\r\n')
    result = TextReader(StringIO(text), **kwargs).read()
    expected = TextReader(StringIO(nice_text), **kwargs).read()
    assert_array_dicts_equal(result, expected)
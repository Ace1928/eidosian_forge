from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_header_not_first_line(all_parsers):
    parser = all_parsers
    data = 'got,to,ignore,this,line\ngot,to,ignore,this,line\nindex,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\n'
    data2 = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\n'
    result = parser.read_csv(StringIO(data), header=2, index_col=0)
    expected = parser.read_csv(StringIO(data2), header=0, index_col=0)
    tm.assert_frame_equal(result, expected)
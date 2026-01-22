from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('kwargs', [{'header': [0, 1]}, {'skiprows': 2, 'names': [('a', 'q'), ('a', 'r'), ('a', 's'), ('b', 't'), ('c', 'u'), ('c', 'v')]}, {'skiprows': 2, 'names': [_TestTuple('a', 'q'), _TestTuple('a', 'r'), _TestTuple('a', 's'), _TestTuple('b', 't'), _TestTuple('c', 'u'), _TestTuple('c', 'v')]}])
def test_header_multi_index_common_format3(all_parsers, kwargs):
    parser = all_parsers
    expected = DataFrame([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], index=['one', 'two'], columns=MultiIndex.from_tuples([('a', 'q'), ('a', 'r'), ('a', 's'), ('b', 't'), ('c', 'u'), ('c', 'v')]))
    expected = expected.reset_index(drop=True)
    data = 'a,a,a,b,c,c\nq,r,s,t,u,v\n1,2,3,4,5,6\n7,8,9,10,11,12'
    result = parser.read_csv(StringIO(data), index_col=None, **kwargs)
    tm.assert_frame_equal(result, expected)
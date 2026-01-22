from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_textreader_obj(self):
    data = 'index,A,B,C,D\n                  foo,2,3,4,5\n                  bar,7,8,9,10\n                  baz,12,13,14,15\n                  qux,12,13,14,15\n                  foo2,12,13,14,15\n                  bar2,12,13,14,15\n               '
    with read_csv(StringIO(data), chunksize=1) as reader:
        result = concat(reader, ignore_index=True)
    expected = read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_empty_with_dup_column_pass_dtype_by_indexes(all_parsers):
    parser = all_parsers
    expected = concat([Series([], name='one', dtype='u1'), Series([], name='one.1', dtype='f')], axis=1)
    data = 'one,one'
    result = parser.read_csv(StringIO(data), dtype={0: 'u1', 1: 'f'})
    tm.assert_frame_equal(result, expected)
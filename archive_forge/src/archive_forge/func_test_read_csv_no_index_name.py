from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
def test_read_csv_no_index_name(all_parsers, csv_dir_path):
    parser = all_parsers
    csv2 = os.path.join(csv_dir_path, 'test2.csv')
    result = parser.read_csv(csv2, index_col=0, parse_dates=True)
    expected = DataFrame([[0.980269, 3.685731, -0.364216805298, -1.159738, 'foo'], [1.047916, -0.041232, -0.16181208307, 0.212549, 'bar'], [0.498581, 0.731168, -0.537677223318, 1.34627, 'baz'], [1.120202, 1.567621, 0.00364077397681, 0.675253, 'qux'], [-0.487094, 0.571455, -1.6116394093, 0.103469, 'foo2']], columns=['A', 'B', 'C', 'D', 'E'], index=Index([datetime(2000, 1, 3), datetime(2000, 1, 4), datetime(2000, 1, 5), datetime(2000, 1, 6), datetime(2000, 1, 7)]))
    tm.assert_frame_equal(result, expected)
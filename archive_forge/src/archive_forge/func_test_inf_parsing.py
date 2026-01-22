from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('na_filter', [True, False])
def test_inf_parsing(all_parsers, na_filter):
    parser = all_parsers
    data = ',A\na,inf\nb,-inf\nc,+Inf\nd,-Inf\ne,INF\nf,-INF\ng,+INf\nh,-INf\ni,inF\nj,-inF'
    expected = DataFrame({'A': [float('inf'), float('-inf')] * 5}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    result = parser.read_csv(StringIO(data), index_col=0, na_filter=na_filter)
    tm.assert_frame_equal(result, expected)
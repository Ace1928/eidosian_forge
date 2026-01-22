from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_no_na_values_no_keep_default(all_parsers):
    data = 'A,B,C\na,1,None\nb,2,two\n,3,None\nd,4,nan\ne,5,five\nnan,6,\ng,7,seven\n'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), keep_default_na=False)
    expected = DataFrame({'A': ['a', 'b', '', 'd', 'e', 'nan', 'g'], 'B': [1, 2, 3, 4, 5, 6, 7], 'C': ['None', 'two', 'None', 'nan', 'five', '', 'seven']})
    tm.assert_frame_equal(result, expected)
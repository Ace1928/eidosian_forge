from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_header_delim_whitespace(all_parsers):
    parser = all_parsers
    data = 'a,b\n1,2\n3,4\n    '
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), delim_whitespace=True)
    expected = DataFrame({'a,b': ['1,2', '3,4']})
    tm.assert_frame_equal(result, expected)
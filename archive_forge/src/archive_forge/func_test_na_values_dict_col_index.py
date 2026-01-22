from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_na_values_dict_col_index(all_parsers):
    data = 'a\nfoo\n1'
    parser = all_parsers
    na_values = {0: 'foo'}
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values)
        return
    result = parser.read_csv(StringIO(data), na_values=na_values)
    expected = DataFrame({'a': [np.nan, 1]})
    tm.assert_frame_equal(result, expected)
from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_values,row_data', [(1, [[np.nan, 2.0], [2.0, np.nan]]), ({'a': 2, 'b': 1}, [[1.0, 2.0], [np.nan, np.nan]])])
def test_na_values_scalar(all_parsers, na_values, row_data):
    parser = all_parsers
    names = ['a', 'b']
    data = '1,2\n2,1'
    if parser.engine == 'pyarrow' and isinstance(na_values, dict):
        if isinstance(na_values, dict):
            err = ValueError
            msg = "The pyarrow engine doesn't support passing a dict for na_values"
        else:
            err = TypeError
            msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    elif parser.engine == 'pyarrow':
        msg = "The 'pyarrow' engine requires all na_values to be strings"
        with pytest.raises(TypeError, match=msg):
            parser.read_csv(StringIO(data), names=names, na_values=na_values)
        return
    result = parser.read_csv(StringIO(data), names=names, na_values=na_values)
    expected = DataFrame(row_data, columns=names)
    tm.assert_frame_equal(result, expected)
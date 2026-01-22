from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_values', ['baz', ['baz']])
def test_custom_na_values(all_parsers, na_values):
    parser = all_parsers
    data = 'A,B,C\nignore,this,row\n1,NA,3\n-1.#IND,5,baz\n7,8,NaN\n'
    expected = DataFrame([[1.0, np.nan, 3], [np.nan, 5, np.nan], [7, 8, np.nan]], columns=['A', 'B', 'C'])
    if parser.engine == 'pyarrow':
        msg = "skiprows argument must be an integer when using engine='pyarrow'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
        return
    result = parser.read_csv(StringIO(data), na_values=na_values, skiprows=[1])
    tm.assert_frame_equal(result, expected)
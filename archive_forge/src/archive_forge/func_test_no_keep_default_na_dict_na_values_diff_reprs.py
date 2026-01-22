from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('col_zero_na_values', [113125, '113125'])
def test_no_keep_default_na_dict_na_values_diff_reprs(all_parsers, col_zero_na_values):
    data = '113125,"blah","/blaha",kjsdkj,412.166,225.874,214.008\n729639,"qwer","",asdfkj,466.681,,252.373\n'
    parser = all_parsers
    expected = DataFrame({0: [np.nan, 729639.0], 1: [np.nan, 'qwer'], 2: ['/blaha', np.nan], 3: ['kjsdkj', 'asdfkj'], 4: [412.166, 466.681], 5: ['225.874', ''], 6: [np.nan, 252.373]})
    if parser.engine == 'pyarrow':
        msg = "The pyarrow engine doesn't support passing a dict for na_values"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=None, keep_default_na=False, na_values={2: '', 6: '214.008', 1: 'blah', 0: col_zero_na_values})
        return
    result = parser.read_csv(StringIO(data), header=None, keep_default_na=False, na_values={2: '', 6: '214.008', 1: 'blah', 0: col_zero_na_values})
    tm.assert_frame_equal(result, expected)
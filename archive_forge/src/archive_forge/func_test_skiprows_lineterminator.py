from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('lineterminator', ['\n', '\r\n', '\r'])
def test_skiprows_lineterminator(all_parsers, lineterminator, request):
    parser = all_parsers
    data = '\n'.join(['SMOSMANIA ThetaProbe-ML2X ', '2007/01/01 01:00   0.2140 U M ', '2007/01/01 02:00   0.2141 M O ', '2007/01/01 04:00   0.2142 D M '])
    expected = DataFrame([['2007/01/01', '01:00', 0.214, 'U', 'M'], ['2007/01/01', '02:00', 0.2141, 'M', 'O'], ['2007/01/01', '04:00', 0.2142, 'D', 'M']], columns=['date', 'time', 'var', 'flag', 'oflag'])
    if parser.engine == 'python' and lineterminator == '\r':
        mark = pytest.mark.xfail(reason="'CR' not respect with the Python parser yet")
        request.applymarker(mark)
    data = data.replace('\n', lineterminator)
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg, check_stacklevel=False):
        result = parser.read_csv(StringIO(data), skiprows=1, delim_whitespace=True, names=['date', 'time', 'var', 'flag', 'oflag'])
    tm.assert_frame_equal(result, expected)
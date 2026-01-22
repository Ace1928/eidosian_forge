from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('conv', [None, np.int64, np.uint64])
def test_int64_overflow(all_parsers, conv, request):
    data = 'ID\n00013007854817840016671868\n00013007854817840016749251\n00013007854817840016754630\n00013007854817840016781876\n00013007854817840017028824\n00013007854817840017963235\n00013007854817840018860166'
    parser = all_parsers
    if conv is None:
        if parser.engine == 'pyarrow':
            mark = pytest.mark.xfail(reason='parses to float64')
            request.applymarker(mark)
        result = parser.read_csv(StringIO(data))
        expected = DataFrame(['00013007854817840016671868', '00013007854817840016749251', '00013007854817840016754630', '00013007854817840016781876', '00013007854817840017028824', '00013007854817840017963235', '00013007854817840018860166'], columns=['ID'])
        tm.assert_frame_equal(result, expected)
    else:
        msg = '|'.join(['Python int too large to convert to C long', 'long too big to convert', 'int too big to convert'])
        err = OverflowError
        if parser.engine == 'pyarrow':
            err = ValueError
            msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(err, match=msg):
            parser.read_csv(StringIO(data), converters={'ID': conv})
from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import (
import pandas._testing as tm
def test_chunksize_second_block_shorter(all_parsers):
    parser = all_parsers
    data = 'a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11\n'
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=2)
        return
    result_chunks = parser.read_csv(StringIO(data), chunksize=2)
    expected_frames = [DataFrame({'a': [1, 5], 'b': [2, 6], 'c': [3, 7], 'd': [4, 8]}), DataFrame({'a': [9], 'b': [10], 'c': [11], 'd': [np.nan]}, index=[2])]
    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])
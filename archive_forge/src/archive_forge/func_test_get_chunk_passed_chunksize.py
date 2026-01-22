from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import (
import pandas._testing as tm
def test_get_chunk_passed_chunksize(all_parsers):
    parser = all_parsers
    data = 'A,B,C\n1,2,3\n4,5,6\n7,8,9\n1,2,3'
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2) as reader:
                reader.get_chunk()
        return
    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result, expected)
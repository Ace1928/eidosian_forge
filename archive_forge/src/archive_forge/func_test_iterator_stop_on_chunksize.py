from io import StringIO
import pytest
from pandas import (
import pandas._testing as tm
def test_iterator_stop_on_chunksize(all_parsers):
    parser = all_parsers
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6\nbaz,7,8,9\n'
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=1)
        return
    with parser.read_csv(StringIO(data), chunksize=1) as reader:
        result = list(reader)
    assert len(result) == 3
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['foo', 'bar', 'baz'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(concat(result), expected)
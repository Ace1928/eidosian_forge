from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('header', ['infer', None])
def test_comment_first_line(all_parsers, header):
    parser = all_parsers
    data = '# notes\na,b,c\n# more notes\n1,2,3'
    if header is None:
        expected = DataFrame({0: ['a', '1'], 1: ['b', '2'], 2: ['c', '3']})
    else:
        expected = DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
    if parser.engine == 'pyarrow':
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment='#', header=header)
        return
    result = parser.read_csv(StringIO(data), comment='#', header=header)
    tm.assert_frame_equal(result, expected)
from io import StringIO
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_comment_skiprows_header(all_parsers):
    parser = all_parsers
    data = '# empty\n# second empty line\n# third empty line\nX,Y,Z\n1,2,3\nA,B,C\n1,2.,4.\n5.,NaN,10.0\n'
    expected = DataFrame([[1.0, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=['A', 'B', 'C'])
    if parser.engine == 'pyarrow':
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), comment='#', skiprows=4, header=1)
        return
    result = parser.read_csv(StringIO(data), comment='#', skiprows=4, header=1)
    tm.assert_frame_equal(result, expected)
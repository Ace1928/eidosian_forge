from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_converter_multi_index(all_parsers):
    parser = all_parsers
    data = 'A,B,B\nX,Y,Z\n1,2,3'
    if parser.engine == 'pyarrow':
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), header=list(range(2)), converters={('A', 'X'): np.int32, ('B', 'Y'): np.int32, ('B', 'Z'): np.float32})
        return
    result = parser.read_csv(StringIO(data), header=list(range(2)), converters={('A', 'X'): np.int32, ('B', 'Y'): np.int32, ('B', 'Z'): np.float32})
    expected = DataFrame({('A', 'X'): np.int32([1]), ('B', 'Y'): np.int32([2]), ('B', 'Z'): np.float32([3])})
    tm.assert_frame_equal(result, expected)
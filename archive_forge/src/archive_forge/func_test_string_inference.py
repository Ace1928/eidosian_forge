from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_string_inference(all_parsers):
    pytest.importorskip('pyarrow')
    dtype = 'string[pyarrow_numpy]'
    data = 'a,b\nx,1\ny,2\n,3'
    parser = all_parsers
    with pd.option_context('future.infer_string', True):
        result = parser.read_csv(StringIO(data))
    expected = DataFrame({'a': pd.Series(['x', 'y', None], dtype=dtype), 'b': [1, 2, 3]}, columns=pd.Index(['a', 'b'], dtype=dtype))
    tm.assert_frame_equal(result, expected)
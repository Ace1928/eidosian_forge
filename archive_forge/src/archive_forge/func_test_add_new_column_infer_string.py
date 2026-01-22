from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_add_new_column_infer_string():
    pytest.importorskip('pyarrow')
    df = DataFrame({'x': [1]})
    with pd.option_context('future.infer_string', True):
        df.loc[df['x'] == 1, 'y'] = '1'
    expected = DataFrame({'x': [1], 'y': Series(['1'], dtype='string[pyarrow_numpy]')}, columns=Index(['x', 'y'], dtype=object))
    tm.assert_frame_equal(df, expected)
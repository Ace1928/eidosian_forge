from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_coerces_datetime(all_parsers):
    parser = all_parsers
    dti = pd.DatetimeIndex(['2017-01-01', '2018-01-01', '2019-01-01'], freq=None)
    dtype = {'b': CategoricalDtype(dti)}
    data = 'b\n2017-01-01\n2018-01-01\n2019-01-01'
    expected = DataFrame({'b': Categorical(dtype['b'].categories)})
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)
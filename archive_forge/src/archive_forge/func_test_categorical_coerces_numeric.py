from io import StringIO
import os
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_coerces_numeric(all_parsers):
    parser = all_parsers
    dtype = {'b': CategoricalDtype([1, 2, 3])}
    data = 'b\n1\n1\n2\n3'
    expected = DataFrame({'b': Categorical([1, 1, 2, 3])})
    result = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(result, expected)
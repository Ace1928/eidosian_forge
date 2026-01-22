import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_retain_attrs(self, any_numpy_dtype):
    df = DataFrame({'a': [0, 1, 2], 'b': [3, 4, 5]})
    df.attrs['Location'] = 'Michigan'
    result = df.astype({'a': any_numpy_dtype}).attrs
    expected = df.attrs
    tm.assert_dict_equal(expected, result)
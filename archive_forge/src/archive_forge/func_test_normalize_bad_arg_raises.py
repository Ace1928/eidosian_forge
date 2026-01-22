from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_normalize_bad_arg_raises(any_string_dtype):
    ser = Series(['ABC', 'ＡＢＣ', '１２３', np.nan, 'ｱｲｴ'], index=['a', 'b', 'c', 'd', 'e'], dtype=any_string_dtype)
    with pytest.raises(ValueError, match='invalid normalization form'):
        ser.str.normalize('xxx')
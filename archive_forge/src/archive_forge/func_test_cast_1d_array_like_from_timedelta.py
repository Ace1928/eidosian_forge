import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_cast_1d_array_like_from_timedelta():
    td = Timedelta(1)
    res = construct_1d_arraylike_from_scalar(td, 2, np.dtype('m8[ns]'))
    assert res[0] == td
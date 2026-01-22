from datetime import (
from importlib import reload
import string
import sys
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', np.typecodes['All'])
def test_astype_empty_constructor_equality(self, dtype):
    if dtype not in ('S', 'V', 'M', 'm'):
        init_empty = Series([], dtype=dtype)
        as_type_empty = Series([]).astype(dtype)
        tm.assert_series_equal(init_empty, as_type_empty)
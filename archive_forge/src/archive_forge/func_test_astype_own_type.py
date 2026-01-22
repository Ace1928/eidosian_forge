import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.parametrize('copy', [True, False])
def test_astype_own_type(self, data, copy):
    result = data.astype(data.dtype, copy=copy)
    assert (result is data) is (not copy)
    tm.assert_extension_array_equal(result, data)
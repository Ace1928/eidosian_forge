from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('test_input,error_type', [(Series([], dtype='float64'), ValueError), (Series(['foo', 'bar', 'baz']), TypeError), (Series([(1,), (2,)]), TypeError), (Series(['foo', 'foo', 'bar', 'bar', None, np.nan, 'baz']), TypeError)])
def test_assert_idxminmax_empty_raises(self, test_input, error_type):
    """
        Cases where ``Series.argmax`` and related should raise an exception
        """
    test_input = Series([], dtype='float64')
    msg = 'attempt to get argmin of an empty sequence'
    with pytest.raises(ValueError, match=msg):
        test_input.idxmin()
    with pytest.raises(ValueError, match=msg):
        test_input.idxmin(skipna=False)
    msg = 'attempt to get argmax of an empty sequence'
    with pytest.raises(ValueError, match=msg):
        test_input.idxmax()
    with pytest.raises(ValueError, match=msg):
        test_input.idxmax(skipna=False)
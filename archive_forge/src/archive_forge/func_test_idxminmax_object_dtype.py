from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_idxminmax_object_dtype(self, using_infer_string):
    ser = Series(['foo', 'bar', 'baz'])
    assert ser.idxmax() == 0
    assert ser.idxmax(skipna=False) == 0
    assert ser.idxmin() == 1
    assert ser.idxmin(skipna=False) == 1
    ser2 = Series([(1,), (2,)])
    assert ser2.idxmax() == 1
    assert ser2.idxmax(skipna=False) == 1
    assert ser2.idxmin() == 0
    assert ser2.idxmin(skipna=False) == 0
    if not using_infer_string:
        ser3 = Series(['foo', 'foo', 'bar', 'bar', None, np.nan, 'baz'])
        msg = "'>' not supported between instances of 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            ser3.idxmax()
        with pytest.raises(TypeError, match=msg):
            ser3.idxmax(skipna=False)
        msg = "'<' not supported between instances of 'float' and 'str'"
        with pytest.raises(TypeError, match=msg):
            ser3.idxmin()
        with pytest.raises(TypeError, match=msg):
            ser3.idxmin(skipna=False)
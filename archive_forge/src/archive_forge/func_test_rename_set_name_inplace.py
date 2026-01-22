from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_set_name_inplace(self, using_infer_string):
    ser = Series(range(3), index=list('abc'))
    for name in ['foo', 123, 123.0, datetime(2001, 11, 11), ('foo',)]:
        ser.rename(name, inplace=True)
        assert ser.name == name
        exp = np.array(['a', 'b', 'c'], dtype=np.object_)
        if using_infer_string:
            exp = array(exp, dtype='string[pyarrow_numpy]')
            tm.assert_extension_array_equal(ser.index.values, exp)
        else:
            tm.assert_numpy_array_equal(ser.index.values, exp)
from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_rename_set_name(self, using_infer_string):
    ser = Series(range(4), index=list('abcd'))
    for name in ['foo', 123, 123.0, datetime(2001, 11, 11), ('foo',)]:
        result = ser.rename(name)
        assert result.name == name
        if using_infer_string:
            tm.assert_extension_array_equal(result.index.values, ser.index.values)
        else:
            tm.assert_numpy_array_equal(result.index.values, ser.index.values)
        assert ser.name is None
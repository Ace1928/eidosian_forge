import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_xs_key_as_list(self):
    mi = MultiIndex.from_tuples([('a', 'x')], names=['level1', 'level2'])
    ser = Series([1], index=mi)
    with pytest.raises(TypeError, match='list keys are not supported'):
        ser.xs(['a', 'x'], axis=0, drop_level=False)
    with pytest.raises(TypeError, match='list keys are not supported'):
        ser.xs(['a'], axis=0, drop_level=False)
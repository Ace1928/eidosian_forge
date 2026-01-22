from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('args', [Index(['a', 'b']), Series(['a', 'b']), np.array(['a', 'b']), [], np.arange(0, 10), np.array([1]), [1]])
def test_constructor_additional_invalid_args(self, args):
    msg = f'Value needs to be a scalar value, was type {type(args).__name__}'
    with pytest.raises(TypeError, match=msg):
        RangeIndex(args)
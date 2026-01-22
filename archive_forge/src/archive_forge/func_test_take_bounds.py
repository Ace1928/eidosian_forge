import numpy as np
import pytest
from pandas import Categorical
import pandas._testing as tm
def test_take_bounds(self, allow_fill):
    cat = Categorical(['a', 'b', 'a'])
    if allow_fill:
        msg = 'indices are out-of-bounds'
    else:
        msg = 'index 4 is out of bounds for( axis 0 with)? size 3'
    with pytest.raises(IndexError, match=msg):
        cat.take([4, 5], allow_fill=allow_fill)
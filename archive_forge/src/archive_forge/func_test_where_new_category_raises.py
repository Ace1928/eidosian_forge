import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_where_new_category_raises(self):
    ser = Series(Categorical(['a', 'b', 'c']))
    msg = 'Cannot setitem on a Categorical with a new category'
    with pytest.raises(TypeError, match=msg):
        ser.where([True, False, True], 'd')
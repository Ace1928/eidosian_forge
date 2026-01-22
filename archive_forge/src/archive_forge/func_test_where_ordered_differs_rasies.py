import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_where_ordered_differs_rasies(self):
    ser = Series(Categorical(['a', 'b', 'c'], categories=['d', 'c', 'b', 'a'], ordered=True))
    other = Categorical(['b', 'c', 'a'], categories=['a', 'c', 'b', 'd'], ordered=True)
    with pytest.raises(TypeError, match='without identical categories'):
        ser.where([True, False, True], other)
from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
def test_constructor_errors(self, klass):
    ivs = [Interval(0, 1, closed='right'), Interval(2, 3, closed='left')]
    msg = 'intervals must all be closed on the same side'
    with pytest.raises(ValueError, match=msg):
        klass(ivs)
    msg = '(IntervalIndex|Index)\\(...\\) must be called with a collection of some kind, 5 was passed'
    with pytest.raises(TypeError, match=msg):
        klass(5)
    msg = "type <class 'numpy.int(32|64)'> with value 0 is not an interval"
    with pytest.raises(TypeError, match=msg):
        klass([0, 1])
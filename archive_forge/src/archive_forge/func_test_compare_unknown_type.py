from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('val', ['string', 1])
def test_compare_unknown_type(self, val):
    t = Timedelta('1s')
    msg = "not supported between instances of 'Timedelta' and '(int|str)'"
    with pytest.raises(TypeError, match=msg):
        t >= val
    with pytest.raises(TypeError, match=msg):
        t > val
    with pytest.raises(TypeError, match=msg):
        t <= val
    with pytest.raises(TypeError, match=msg):
        t < val
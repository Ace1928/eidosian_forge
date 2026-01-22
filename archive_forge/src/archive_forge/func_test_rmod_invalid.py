from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_rmod_invalid(self):
    td = Timedelta(minutes=3)
    msg = 'unsupported operand'
    with pytest.raises(TypeError, match=msg):
        Timestamp('2018-01-22') % td
    with pytest.raises(TypeError, match=msg):
        15 % td
    with pytest.raises(TypeError, match=msg):
        16.0 % td
    msg = 'Invalid dtype int'
    with pytest.raises(TypeError, match=msg):
        np.array([22, 24]) % td
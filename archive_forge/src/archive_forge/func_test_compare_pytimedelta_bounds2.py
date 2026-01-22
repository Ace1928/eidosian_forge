from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_compare_pytimedelta_bounds2(self):
    pytd = timedelta(days=999999999, seconds=86399)
    td64 = np.timedelta64(pytd.days, 'D') + np.timedelta64(pytd.seconds, 's')
    td = Timedelta(td64)
    assert td.days == pytd.days
    assert td.seconds == pytd.seconds
    assert td == pytd
    assert not td != pytd
    assert not td < pytd
    assert not td > pytd
    assert td <= pytd
    assert td >= pytd
    td2 = td - Timedelta(seconds=1).as_unit('s')
    assert td2 != pytd
    assert not td2 == pytd
    assert td2 < pytd
    assert td2 <= pytd
    assert not td2 > pytd
    assert not td2 >= pytd
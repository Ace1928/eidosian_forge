from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_rfloordiv_timedeltalike_scalar(self):
    td = Timedelta(hours=3, minutes=3)
    scalar = Timedelta(hours=3, minutes=4)
    assert td.__rfloordiv__(scalar) == 1
    assert (-td).__rfloordiv__(scalar.to_pytimedelta()) == -2
    assert (2 * td).__rfloordiv__(scalar.to_timedelta64()) == 0
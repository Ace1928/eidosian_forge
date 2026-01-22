from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('op', [operator.add, ops.radd])
def test_td_add_offset(self, op):
    td = Timedelta(10, unit='d')
    result = op(td, offsets.Hour(6))
    assert isinstance(result, Timedelta)
    assert result == Timedelta(days=10, hours=6)
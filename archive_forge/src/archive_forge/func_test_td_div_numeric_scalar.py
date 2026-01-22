from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_td_div_numeric_scalar(self):
    td = Timedelta(10, unit='d')
    result = td / 2
    assert isinstance(result, Timedelta)
    assert result == Timedelta(days=5)
    result = td / 5
    assert isinstance(result, Timedelta)
    assert result == Timedelta(days=2)
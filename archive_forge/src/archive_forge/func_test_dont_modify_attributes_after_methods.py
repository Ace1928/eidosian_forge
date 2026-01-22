import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_dont_modify_attributes_after_methods(arithmetic_win_operators, closed, center, min_periods, step):
    roll_obj = Series(range(1)).rolling(1, center=center, closed=closed, min_periods=min_periods, step=step)
    expected = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    getattr(roll_obj, arithmetic_win_operators)()
    result = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    assert result == expected
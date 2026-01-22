from copy import deepcopy
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', ['_set_axis_name', 'rename_axis'])
def test_set_axis_name(self, func):
    df = DataFrame([[1, 2], [3, 4]])
    result = methodcaller(func, 'foo')(df)
    assert df.index.name is None
    assert result.index.name == 'foo'
    result = methodcaller(func, 'cols', axis=1)(df)
    assert df.columns.name is None
    assert result.columns.name == 'cols'
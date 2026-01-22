import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('attr', ['index', 'columns'])
def test_copy_index_name_checking(self, float_frame, attr):
    ind = getattr(float_frame, attr)
    ind.name = None
    cp = float_frame.copy()
    getattr(cp, attr).name = 'foo'
    assert getattr(float_frame, attr).name is None
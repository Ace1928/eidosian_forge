import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index,val', [(Index([0, 1, '2']), 0), (Index([0, 1, '2']), '2')])
def test_mixed_index_contains(self, index, val):
    assert val in index
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
def test_get_loc_non_hashable(self, index):
    with pytest.raises(InvalidIndexError, match='[0, 1]'):
        index.get_loc([0, 1])
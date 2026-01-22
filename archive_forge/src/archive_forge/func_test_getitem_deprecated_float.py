import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas.core.dtypes.common import (
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx', [Index([1, 2, 3]), Index([0.1, 0.2, 0.3]), Index(['a', 'b', 'c'])])
def test_getitem_deprecated_float(idx):
    msg = 'Indexing with a float is no longer supported'
    with pytest.raises(IndexError, match=msg):
        idx[1.0]
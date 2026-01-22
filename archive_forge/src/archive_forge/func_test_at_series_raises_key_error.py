from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_series_raises_key_error(self, indexer_al):
    ser = Series([1, 2, 3], index=[3, 2, 1])
    result = indexer_al(ser)[1]
    assert result == 3
    with pytest.raises(KeyError, match='a'):
        indexer_al(ser)['a']
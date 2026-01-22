import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_win_type_not_implemented():
    pytest.importorskip('scipy')

    class CustomIndexer(BaseIndexer):

        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return (np.array([0, 1]), np.array([1, 2]))
    df = DataFrame({'values': range(2)})
    indexer = CustomIndexer()
    with pytest.raises(NotImplementedError, match='BaseIndexer subclasses not'):
        df.rolling(indexer, win_type='boxcar')
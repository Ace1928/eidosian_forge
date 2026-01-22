import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_non_matching(self, series_with_interval_index, indexer_sl):
    ser = series_with_interval_index.copy()
    with pytest.raises(KeyError, match='\\[-1\\] not in index'):
        indexer_sl(ser)[[-1, 3, 4, 5]]
    with pytest.raises(KeyError, match='\\[-1\\] not in index'):
        indexer_sl(ser)[[-1, 3]]
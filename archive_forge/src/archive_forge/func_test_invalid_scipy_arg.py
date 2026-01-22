import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_invalid_scipy_arg():
    pytest.importorskip('scipy')
    msg = 'boxcar\\(\\) got an unexpected'
    with pytest.raises(TypeError, match=msg):
        Series(range(3)).rolling(1, win_type='boxcar').mean(foo='bar')
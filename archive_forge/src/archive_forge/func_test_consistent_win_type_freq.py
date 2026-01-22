import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
@pytest.mark.parametrize('arg', [2000000000, '2s', Timedelta('2s')])
def test_consistent_win_type_freq(arg):
    pytest.importorskip('scipy')
    s = Series(range(1))
    with pytest.raises(ValueError, match='Invalid win_type freq'):
        s.rolling(arg, win_type='freq')
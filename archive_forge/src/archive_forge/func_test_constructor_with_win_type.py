import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
def test_constructor_with_win_type(frame_or_series, win_types):
    pytest.importorskip('scipy')
    c = frame_or_series(range(5)).rolling
    c(win_type=win_types, window=2)
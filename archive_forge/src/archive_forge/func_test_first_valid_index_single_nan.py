import numpy as np
import pytest
from pandas import (
def test_first_valid_index_single_nan(self, frame_or_series):
    obj = frame_or_series([np.nan])
    assert obj.first_valid_index() is None
    assert obj.iloc[:0].first_valid_index() is None
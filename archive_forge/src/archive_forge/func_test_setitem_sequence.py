import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_sequence(self, data, box_in_series):
    if box_in_series:
        data = pd.Series(data)
    original = data.copy()
    data[[0, 1]] = [data[1], data[0]]
    assert data[0] == original[1]
    assert data[1] == original[0]
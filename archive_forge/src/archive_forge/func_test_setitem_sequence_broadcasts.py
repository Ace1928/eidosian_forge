import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_setitem_sequence_broadcasts(self, data, box_in_series):
    if box_in_series:
        data = pd.Series(data)
    data[[0, 1]] = data[2]
    assert data[0] == data[2]
    assert data[1] == data[2]
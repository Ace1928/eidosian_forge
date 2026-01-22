import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_getitem_empty(self, data):
    result = data[[]]
    assert len(result) == 0
    assert isinstance(result, type(data))
    expected = data[np.array([], dtype='int64')]
    tm.assert_extension_array_equal(result, expected)
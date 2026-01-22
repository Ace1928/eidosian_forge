from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data, to_replace, value, expected', [([1], [1.0], [0], [0]), ([1], [1], [0], [0]), ([1.0], [1.0], [0], [0.0]), ([1.0], [1], [0], [0.0])])
@pytest.mark.parametrize('box', [list, tuple, np.array])
def test_replace_list_with_mixed_type(self, data, to_replace, value, expected, box, frame_or_series):
    obj = frame_or_series(data)
    expected = frame_or_series(expected)
    result = obj.replace(box(to_replace), value)
    tm.assert_equal(result, expected)
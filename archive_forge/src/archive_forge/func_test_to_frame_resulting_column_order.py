import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_frame_resulting_column_order():
    expected = ['z', 0, 'a']
    mi = MultiIndex.from_arrays([['a', 'b', 'c'], ['x', 'y', 'z'], ['q', 'w', 'e']], names=expected)
    result = mi.to_frame().columns.tolist()
    assert result == expected
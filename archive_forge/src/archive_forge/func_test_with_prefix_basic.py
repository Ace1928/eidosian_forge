import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_with_prefix_basic(dummies_basic):
    expected = DataFrame({'col1': ['a', 'b', 'a'], 'col2': ['b', 'a', 'c']})
    result = from_dummies(dummies_basic, sep='_')
    tm.assert_frame_equal(result, expected)
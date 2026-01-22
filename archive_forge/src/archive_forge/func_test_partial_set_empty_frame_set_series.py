import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_partial_set_empty_frame_set_series(self):
    df = DataFrame(Series(dtype=object))
    expected = DataFrame({0: Series(dtype=object)})
    tm.assert_frame_equal(df, expected)
    df = DataFrame(Series(name='foo', dtype=object))
    expected = DataFrame({'foo': Series(dtype=object)})
    tm.assert_frame_equal(df, expected)
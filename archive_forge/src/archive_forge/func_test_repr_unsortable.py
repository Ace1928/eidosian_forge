from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_unsortable(self, float_frame):
    unsortable = DataFrame({'foo': [1] * 50, datetime.today(): [1] * 50, 'bar': ['bar'] * 50, datetime.today() + timedelta(1): ['bar'] * 50}, index=np.arange(50))
    repr(unsortable)
    fmt.set_option('display.precision', 3)
    repr(float_frame)
    fmt.set_option('display.max_rows', 10, 'display.max_columns', 2)
    repr(float_frame)
    fmt.set_option('display.max_rows', 1000, 'display.max_columns', 1000)
    repr(float_frame)
    tm.reset_display_options()
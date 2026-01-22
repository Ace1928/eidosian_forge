from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_repr_float_frame_options(self, float_frame):
    repr(float_frame)
    with option_context('display.precision', 3):
        repr(float_frame)
    with option_context('display.max_rows', 10, 'display.max_columns', 2):
        repr(float_frame)
    with option_context('display.max_rows', 1000, 'display.max_columns', 1000):
        repr(float_frame)
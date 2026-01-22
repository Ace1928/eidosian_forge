from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_pure_bool_with_string_no_op(self):
    df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
    result = df.replace('asdf', 'fdsa')
    tm.assert_frame_equal(df, result)
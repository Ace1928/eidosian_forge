from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_bytes(self, frame_or_series):
    obj = frame_or_series(['o']).astype('|S')
    expected = obj.copy()
    obj = obj.replace({None: np.nan})
    tm.assert_equal(obj, expected)
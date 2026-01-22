from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set float into string")
def test_regex_replace_dict_nested_gh4115(self):
    df = DataFrame({'Type': ['Q', 'T', 'Q', 'Q', 'T'], 'tmp': 2})
    expected = DataFrame({'Type': [0, 1, 0, 0, 1], 'tmp': 2})
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.replace({'Type': {'Q': 0, 'T': 1}})
    tm.assert_frame_equal(result, expected)
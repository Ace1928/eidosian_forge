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
def test_regex_replace_str_to_numeric(self, mix_abc):
    df = DataFrame(mix_abc)
    res = df.replace('\\s*\\.\\s*', 0, regex=True)
    res2 = df.copy()
    return_value = res2.replace('\\s*\\.\\s*', 0, inplace=True, regex=True)
    assert return_value is None
    res3 = df.copy()
    return_value = res3.replace(regex='\\s*\\.\\s*', value=0, inplace=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', 0, 0], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    tm.assert_frame_equal(res3, expec)
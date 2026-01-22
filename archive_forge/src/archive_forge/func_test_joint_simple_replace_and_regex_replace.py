from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('to_replace', [{'': np.nan, ',': ''}, {',': '', '': np.nan}])
def test_joint_simple_replace_and_regex_replace(self, to_replace):
    df = DataFrame({'col1': ['1,000', 'a', '3'], 'col2': ['a', '', 'b'], 'col3': ['a', 'b', 'c']})
    result = df.replace(regex=to_replace)
    expected = DataFrame({'col1': ['1000', 'a', '3'], 'col2': ['a', np.nan, 'b'], 'col3': ['a', 'b', 'c']})
    tm.assert_frame_equal(result, expected)
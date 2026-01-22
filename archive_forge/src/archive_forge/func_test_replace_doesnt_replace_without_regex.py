from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_doesnt_replace_without_regex(self):
    df = DataFrame({'fol': [1, 2, 2, 3], 'T_opp': ['0', 'vr', '0', '0'], 'T_Dir': ['0', '0', '0', 'bt'], 'T_Enh': ['vo', '0', '0', '0']})
    res = df.replace({'\\D': 1})
    tm.assert_frame_equal(df, res)
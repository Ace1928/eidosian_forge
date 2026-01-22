from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_flex_comparison_nat(self):
    df = DataFrame([pd.NaT])
    result = df == pd.NaT
    assert result.iloc[0, 0].item() is False
    result = df.eq(pd.NaT)
    assert result.iloc[0, 0].item() is False
    result = df != pd.NaT
    assert result.iloc[0, 0].item() is True
    result = df.ne(pd.NaT)
    assert result.iloc[0, 0].item() is True
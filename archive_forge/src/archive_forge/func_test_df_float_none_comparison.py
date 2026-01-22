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
def test_df_float_none_comparison(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((8, 3)), index=range(8), columns=['A', 'B', 'C'])
    result = df.__eq__(None)
    assert not result.any().any()
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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't compare string and int")
def test_mixed_comparison(self):
    df = DataFrame([['1989-08-01', 1], ['1989-08-01', 2]])
    other = DataFrame([['a', 'b'], ['c', 'd']])
    result = df == other
    assert not result.any().any()
    result = df != other
    assert result.all().all()
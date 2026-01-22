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
@pytest.mark.parametrize('dim', range(3, 6))
def test_arith_flex_frame_raise(self, all_arithmetic_operators, float_frame, dim):
    op = all_arithmetic_operators
    arr = np.ones((1,) * dim)
    msg = 'Unable to coerce to Series/DataFrame'
    with pytest.raises(ValueError, match=msg):
        getattr(float_frame, op)(arr)
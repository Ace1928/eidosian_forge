from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('op', [operator.lt, operator.gt])
def test_nan_comparison_same_object(op):
    idx = Index([np.nan])
    expected = np.array([False])
    result = op(idx, idx)
    tm.assert_numpy_array_equal(result, expected)
    result = op(idx, idx.copy())
    tm.assert_numpy_array_equal(result, expected)
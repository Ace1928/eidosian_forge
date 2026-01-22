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
@pytest.mark.parametrize('values', [[1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0], [True, True, True, True], ['foo', 'bar', 'baz', 'qux'], date_range('2018-01-01', freq='D', periods=4)])
def test_boolean_cmp(self, values):
    index = Index(values)
    result = index == values
    expected = np.array([True, True, True, True], dtype=bool)
    tm.assert_numpy_array_equal(result, expected)
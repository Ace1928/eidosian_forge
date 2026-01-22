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
@pytest.mark.parametrize('empty', [[], Series(dtype=object), np.array([])])
def test_isin_empty(self, empty):
    index = Index(['a', 'b'])
    expected = np.array([False, False])
    result = index.isin(empty)
    tm.assert_numpy_array_equal(expected, result)
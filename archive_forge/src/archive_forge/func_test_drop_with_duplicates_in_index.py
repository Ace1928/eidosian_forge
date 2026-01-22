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
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_drop_with_duplicates_in_index(self, index):
    if len(index) == 0 or isinstance(index, MultiIndex):
        pytest.skip("Test doesn't make sense for empty MultiIndex")
    if isinstance(index, IntervalIndex) and (not IS64):
        pytest.skip('Cannot test IntervalIndex with int64 dtype on 32 bit platform')
    index = index.unique().repeat(2)
    expected = index[2:]
    result = index.drop(index[0])
    tm.assert_index_equal(result, expected)
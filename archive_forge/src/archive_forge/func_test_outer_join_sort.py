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
def test_outer_join_sort(self):
    left_index = Index(np.random.default_rng(2).permutation(15))
    right_index = date_range('2020-01-01', periods=10)
    with tm.assert_produces_warning(RuntimeWarning):
        result = left_index.join(right_index, how='outer')
    with tm.assert_produces_warning(RuntimeWarning):
        expected = left_index.astype(object).union(right_index.astype(object))
    tm.assert_index_equal(result, expected)
from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
def test_int64_overflow_groupby_large_range(self):
    values = range(55109)
    data = DataFrame.from_dict({'a': values, 'b': values, 'c': values, 'd': values})
    grouped = data.groupby(['a', 'b', 'c', 'd'])
    assert len(grouped) == len(values)
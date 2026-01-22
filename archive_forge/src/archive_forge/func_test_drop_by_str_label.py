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
@pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'float64', 'float32'], indirect=True)
def test_drop_by_str_label(self, index):
    n = len(index)
    drop = index[list(range(5, 10))]
    dropped = index.drop(drop)
    expected = index[list(range(5)) + list(range(10, n))]
    tm.assert_index_equal(dropped, expected)
    dropped = index.drop(index[0])
    expected = index[1:]
    tm.assert_index_equal(dropped, expected)
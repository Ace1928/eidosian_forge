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
def test_map_with_tuples(self):
    index = Index(np.arange(3), dtype=np.int64)
    result = index.map(lambda x: (x,))
    expected = Index([(i,) for i in index])
    tm.assert_index_equal(result, expected)
    result = index.map(lambda x: (x, x == 1))
    expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
    tm.assert_index_equal(result, expected)
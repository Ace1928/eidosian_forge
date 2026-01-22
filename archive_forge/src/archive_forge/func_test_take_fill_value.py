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
def test_take_fill_value(self):
    index = Index(list('ABC'), name='xxx')
    result = index.take(np.array([1, 0, -1]))
    expected = Index(list('BAC'), name='xxx')
    tm.assert_index_equal(result, expected)
    result = index.take(np.array([1, 0, -1]), fill_value=True)
    expected = Index(['B', 'A', np.nan], name='xxx')
    tm.assert_index_equal(result, expected)
    result = index.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
    expected = Index(['B', 'A', 'C'], name='xxx')
    tm.assert_index_equal(result, expected)
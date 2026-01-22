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
@pytest.mark.parametrize('how', ['any', 'all'])
@pytest.mark.parametrize('dtype', [None, object, 'category'])
@pytest.mark.parametrize('vals,expected', [([1, 2, 3], [1, 2, 3]), ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]), (['A', 'B', 'C'], ['A', 'B', 'C']), (['A', np.nan, 'B', 'C'], ['A', 'B', 'C'])])
def test_dropna(self, how, dtype, vals, expected):
    index = Index(vals, dtype=dtype)
    result = index.dropna(how=how)
    expected = Index(expected, dtype=dtype)
    tm.assert_index_equal(result, expected)
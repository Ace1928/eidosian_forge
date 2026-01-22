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
@pytest.mark.filterwarnings('ignore:elementwise comparison failed:FutureWarning')
def test_index_with_tuple_bool(self):
    idx = Index([('a', 'b'), ('b', 'c'), ('c', 'a')])
    result = idx == ('c', 'a')
    expected = np.array([False, False, True])
    tm.assert_numpy_array_equal(result, expected)
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
@pytest.mark.parametrize('mapper', [Series(['foo', 2.0, 'baz'], index=[0, 2, -1]), {0: 'foo', 2: 2.0, -1: 'baz'}])
def test_map_with_non_function_missing_values(self, mapper):
    expected = Index([2.0, np.nan, 'foo'])
    result = Index([2, 1, 0]).map(mapper)
    tm.assert_index_equal(expected, result)
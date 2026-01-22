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
@pytest.mark.parametrize('index', [date_range('2020-01-01', freq='D', periods=10), period_range('2020-01-01', freq='D', periods=10), timedelta_range('1 day', periods=10)])
def test_map_tseries_indices_return_index(self, index):
    expected = Index([1] * 10)
    result = index.map(lambda x: 1)
    tm.assert_index_equal(expected, result)
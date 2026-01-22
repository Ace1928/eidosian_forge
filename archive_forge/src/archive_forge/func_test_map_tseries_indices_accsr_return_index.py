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
def test_map_tseries_indices_accsr_return_index(self):
    date_index = DatetimeIndex(date_range('2020-01-01', periods=24, freq='h'), name='hourly')
    result = date_index.map(lambda x: x.hour)
    expected = Index(np.arange(24, dtype='int64'), name='hourly')
    tm.assert_index_equal(result, expected, exact=True)
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
@pytest.mark.parametrize('attr', ['values', 'asi8'])
@pytest.mark.parametrize('klass', [Index, TimedeltaIndex])
def test_constructor_dtypes_timedelta(self, attr, klass):
    index = timedelta_range('1 days', periods=5)
    index = index._with_freq(None)
    dtype = index.dtype
    values = getattr(index, attr)
    result = klass(values, dtype=dtype)
    tm.assert_index_equal(result, index)
    result = klass(list(values), dtype=dtype)
    tm.assert_index_equal(result, index)
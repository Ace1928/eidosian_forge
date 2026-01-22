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
@pytest.mark.parametrize('index,has_tz', [(date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern'), True), (timedelta_range('1 days', freq='D', periods=3), False), (period_range('2015-01-01', freq='D', periods=3), False)])
def test_constructor_from_series_dtlike(self, index, has_tz):
    result = Index(Series(index))
    tm.assert_index_equal(result, index)
    if has_tz:
        assert result.tz == index.tz
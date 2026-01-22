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
def test_constructor_from_series_freq(self):
    dts = ['1-1-1990', '2-1-1990', '3-1-1990', '4-1-1990', '5-1-1990']
    expected = DatetimeIndex(dts, freq='MS')
    s = Series(pd.to_datetime(dts))
    result = DatetimeIndex(s, freq='MS')
    tm.assert_index_equal(result, expected)
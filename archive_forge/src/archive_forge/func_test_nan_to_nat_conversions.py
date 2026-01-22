import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_nan_to_nat_conversions():
    df = DataFrame({'A': np.asarray(range(10), dtype='float64'), 'B': Timestamp('20010101')})
    df.iloc[3:6, :] = np.nan
    result = df.loc[4, 'B']
    assert result is pd.NaT
    s = df['B'].copy()
    s[8:9] = np.nan
    assert s[8] is pd.NaT
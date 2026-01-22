import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_period_dict(self):
    a = pd.PeriodIndex(['2012-01', 'NaT', '2012-04'], freq='M')
    b = pd.PeriodIndex(['2012-02-01', '2012-03-01', 'NaT'], freq='D')
    df = DataFrame({'a': a, 'b': b})
    assert df['a'].dtype == a.dtype
    assert df['b'].dtype == b.dtype
    df = DataFrame({'a': a.astype(object).tolist(), 'b': b.astype(object).tolist()})
    assert df['a'].dtype == a.dtype
    assert df['b'].dtype == b.dtype
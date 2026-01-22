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
def test_frame_ctor_datetime64_column(self):
    rng = date_range('1/1/2000 00:00:00', '1/1/2000 1:59:50', freq='10s')
    dates = np.asarray(rng)
    df = DataFrame({'A': np.random.default_rng(2).standard_normal(len(rng)), 'B': dates})
    assert np.issubdtype(df['B'].dtype, np.dtype('M8[ns]'))
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
def test_constructor_dtype_nocast_view_dataframe(self, using_copy_on_write, warn_copy_on_write):
    df = DataFrame([[1, 2]])
    should_be_view = DataFrame(df, dtype=df[0].dtype)
    if using_copy_on_write:
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1
    else:
        with tm.assert_cow_warning(warn_copy_on_write):
            should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 99
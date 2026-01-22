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
def test_constructor_manager_resize(self, float_frame):
    index = list(float_frame.index[:5])
    columns = list(float_frame.columns[:3])
    msg = 'Passing a BlockManager to DataFrame'
    with tm.assert_produces_warning(DeprecationWarning, match=msg, check_stacklevel=False):
        result = DataFrame(float_frame._mgr, index=index, columns=columns)
    tm.assert_index_equal(result.index, Index(index))
    tm.assert_index_equal(result.columns, Index(columns))
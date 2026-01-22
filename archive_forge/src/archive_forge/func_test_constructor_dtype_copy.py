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
def test_constructor_dtype_copy(self):
    orig_df = DataFrame({'col1': [1.0], 'col2': [2.0], 'col3': [3.0]})
    new_df = DataFrame(orig_df, dtype=float, copy=True)
    new_df['col1'] = 200.0
    assert orig_df['col1'][0] == 1.0
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
def test_constructor_mix_series_nonseries(self, float_frame):
    df = DataFrame({'A': float_frame['A'], 'B': list(float_frame['B'])}, columns=['A', 'B'])
    tm.assert_frame_equal(df, float_frame.loc[:, ['A', 'B']])
    msg = 'does not match index length'
    with pytest.raises(ValueError, match=msg):
        DataFrame({'A': float_frame['A'], 'B': list(float_frame['B'])[:-2]})
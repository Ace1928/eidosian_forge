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
def test_frame_string_inference_array_string_dtype(self):
    pytest.importorskip('pyarrow')
    dtype = 'string[pyarrow_numpy]'
    expected = DataFrame({'a': ['a', 'b']}, dtype=dtype, columns=Index(['a'], dtype=dtype))
    with pd.option_context('future.infer_string', True):
        df = DataFrame({'a': np.array(['a', 'b'])})
    tm.assert_frame_equal(df, expected)
    expected = DataFrame({0: ['a', 'b'], 1: ['c', 'd']}, dtype=dtype)
    with pd.option_context('future.infer_string', True):
        df = DataFrame(np.array([['a', 'c'], ['b', 'd']]))
    tm.assert_frame_equal(df, expected)
    expected = DataFrame({'a': ['a', 'b'], 'b': ['c', 'd']}, dtype=dtype, columns=Index(['a', 'b'], dtype=dtype))
    with pd.option_context('future.infer_string', True):
        df = DataFrame(np.array([['a', 'c'], ['b', 'd']]), columns=['a', 'b'])
    tm.assert_frame_equal(df, expected)
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
def test_construction_from_ndarray_with_eadtype_mismatched_columns(self):
    arr = np.random.default_rng(2).standard_normal((10, 2))
    dtype = pd.array([2.0]).dtype
    msg = 'len\\(arrays\\) must match len\\(columns\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(arr, columns=['foo'], dtype=dtype)
    arr2 = pd.array([2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=msg):
        DataFrame(arr2, columns=['foo', 'bar'])
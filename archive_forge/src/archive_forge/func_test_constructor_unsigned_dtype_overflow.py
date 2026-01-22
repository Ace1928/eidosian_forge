from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_unsigned_dtype_overflow(self, any_unsigned_int_numpy_dtype):
    if np_version_gt2:
        msg = f'The elements provided in the data cannot all be casted to the dtype {any_unsigned_int_numpy_dtype}'
    else:
        msg = 'Trying to coerce negative values to unsigned integers'
    with pytest.raises(OverflowError, match=msg):
        Series([-1], dtype=any_unsigned_int_numpy_dtype)
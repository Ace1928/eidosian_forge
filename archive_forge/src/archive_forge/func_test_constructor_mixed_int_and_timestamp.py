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
def test_constructor_mixed_int_and_timestamp(self, frame_or_series):
    objs = [Timestamp(9), 10, NaT._value]
    result = frame_or_series(objs, dtype='M8[ns]')
    expected = frame_or_series([Timestamp(9), Timestamp(10), NaT])
    tm.assert_equal(result, expected)
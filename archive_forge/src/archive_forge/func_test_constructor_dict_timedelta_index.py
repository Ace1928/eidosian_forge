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
def test_constructor_dict_timedelta_index(self):
    expected = Series(data=['A', 'B', 'C'], index=pd.to_timedelta([0, 10, 20], unit='s'))
    result = Series(data={pd.to_timedelta(0, unit='s'): 'A', pd.to_timedelta(10, unit='s'): 'B', pd.to_timedelta(20, unit='s'): 'C'}, index=pd.to_timedelta([0, 10, 20], unit='s'))
    tm.assert_series_equal(result, expected)
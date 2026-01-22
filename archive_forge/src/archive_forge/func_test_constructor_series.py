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
def test_constructor_series(self):
    index1 = ['d', 'b', 'a', 'c']
    index2 = sorted(index1)
    s1 = Series([4, 7, -5, 3], index=index1)
    s2 = Series(s1, index=index2)
    tm.assert_series_equal(s2, s1.sort_index())
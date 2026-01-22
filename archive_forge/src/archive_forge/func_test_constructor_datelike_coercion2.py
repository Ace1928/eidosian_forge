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
def test_constructor_datelike_coercion2(self):
    belly = '216 3T19'.split()
    wing1 = '2T15 4H19'.split()
    wing2 = '416 4T20'.split()
    mat = pd.to_datetime('2016-01-22 2019-09-07'.split())
    df = DataFrame({'wing1': wing1, 'wing2': wing2, 'mat': mat}, index=belly)
    result = df.loc['3T19']
    assert result.dtype == object
    result = df.loc['216']
    assert result.dtype == object
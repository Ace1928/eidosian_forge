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
def test_constructor_index_ndim_gt_1_raises(self):
    df = DataFrame([[1, 2], [3, 4], [5, 6]], index=[3, 6, 9])
    with pytest.raises(ValueError, match='Index data must be 1-dimensional'):
        Series([1, 3, 2], index=df)
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
@td.skip_array_manager_invalid_test
def test_from_list_dtype(self):
    result = Series(['1h', '2h'], dtype='timedelta64[ns]')
    assert result._mgr.blocks[0].is_extension is False
    result = Series(['2015'], dtype='datetime64[ns]')
    assert result._mgr.blocks[0].is_extension is False
import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_update_dtype_string(self, ordered):
    dtype = CategoricalDtype(list('abc'), ordered)
    expected_categories = dtype.categories
    expected_ordered = dtype.ordered
    result = dtype.update_dtype('category')
    tm.assert_index_equal(result.categories, expected_categories)
    assert result.ordered is expected_ordered
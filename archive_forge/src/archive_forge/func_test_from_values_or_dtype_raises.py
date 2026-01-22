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
@pytest.mark.parametrize('values, categories, ordered, dtype', [[None, ['a', 'b'], True, dtype2], [None, ['a', 'b'], None, dtype2], [None, None, True, dtype2]])
def test_from_values_or_dtype_raises(self, values, categories, ordered, dtype):
    msg = 'Cannot specify `categories` or `ordered` together with `dtype`.'
    with pytest.raises(ValueError, match=msg):
        CategoricalDtype._from_values_or_dtype(values, categories, ordered, dtype)
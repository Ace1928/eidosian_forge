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
@pytest.mark.parametrize('categories', [list('abc'), None])
@pytest.mark.parametrize('other', ['category', 'not a category'])
def test_categorical_equality_strings(self, categories, ordered, other):
    c1 = CategoricalDtype(categories, ordered)
    result = c1 == other
    expected = other == 'category'
    assert result is expected
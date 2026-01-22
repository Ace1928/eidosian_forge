from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_intersection_name_retention_with_nameless(self, index):
    if isinstance(index, MultiIndex):
        index = index.rename(list(range(index.nlevels)))
    else:
        index = index.rename('foo')
    other = np.asarray(index)
    result = index.intersection(other)
    assert result.name == index.name
    result = index.intersection(other[:0])
    assert result.name == index.name
    result = index[:0].intersection(other)
    assert result.name == index.name
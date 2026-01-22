from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
@pytest.mark.parametrize('klass', [np.array, Series, list])
def test_union_different_type_base(self, klass):
    index = Index([0, 'a', 1, 'b', 2, 'c'])
    first = index[3:]
    second = index[:5]
    result = first.union(klass(second.values))
    assert equal_contents(result, index)
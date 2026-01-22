from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['apply', 'agg', 'transform'])
@pytest.mark.parametrize('func', [{'B': 'sum'}, {'B': ['sum']}])
def test_missing_column(method, func):
    obj = DataFrame({'A': [1]})
    match = re.escape("Column(s) ['B'] do not exist")
    with pytest.raises(KeyError, match=match):
        getattr(obj, method)(func)
from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['apply', 'agg', 'transform'])
@pytest.mark.parametrize('func', [{'A': {'B': 'sum'}}, {'A': {'B': ['sum']}}])
def test_nested_renamer(frame_or_series, method, func):
    obj = frame_or_series({'A': [1]})
    match = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=match):
        getattr(obj, method)(func)
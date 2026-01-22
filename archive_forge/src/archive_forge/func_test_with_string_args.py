from itertools import chain
import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_number
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import (
@pytest.mark.parametrize('arg', ['sum', 'mean', 'min', 'max', 'std'])
def test_with_string_args(datetime_series, arg):
    result = datetime_series.apply(arg)
    expected = getattr(datetime_series, arg)()
    assert result == expected
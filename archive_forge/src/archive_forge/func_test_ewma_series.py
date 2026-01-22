import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('name', ['var', 'std', 'mean'])
def test_ewma_series(series, name):
    series_result = getattr(series.ewm(com=10), name)()
    assert isinstance(series_result, Series)
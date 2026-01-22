import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_sort_index_name(self, datetime_series):
    result = datetime_series.sort_index(ascending=False)
    assert result.name == datetime_series.name
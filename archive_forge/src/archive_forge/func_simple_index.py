import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.indexes.datetimelike import DatetimeLike
@pytest.fixture
def simple_index(self) -> DatetimeIndex:
    return date_range('20130101', periods=5)
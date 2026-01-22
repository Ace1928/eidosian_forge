import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture
def range_index():
    return RangeIndex(3, name='range_index')
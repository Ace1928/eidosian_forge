import pytest
from pandas import (
import pandas._testing as tm
def test_append_empty(self, ci):
    result = ci.append([])
    tm.assert_index_equal(result, ci, exact=True)
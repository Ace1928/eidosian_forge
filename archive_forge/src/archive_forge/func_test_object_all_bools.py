from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_object_all_bools(self):
    arr = np.array([True, False], dtype=object)
    res = Index(arr)
    assert res.dtype == object
    assert Series(arr).dtype == object
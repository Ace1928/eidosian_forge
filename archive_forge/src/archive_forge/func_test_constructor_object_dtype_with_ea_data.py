from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_object_dtype_with_ea_data(self, any_numeric_ea_dtype):
    arr = array([0], dtype=any_numeric_ea_dtype)
    idx = Index(arr, dtype=object)
    assert idx.dtype == object